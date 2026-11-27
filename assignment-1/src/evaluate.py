"""Metrics, cross-validation, bootstrap and residual diagnostics."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, GroupKFold, KFold, cross_val_predict

from .airfoil import FEATURE_SETS, from_model_target, make_pipeline


def metrics(y_true, y_pred) -> dict:
    """Return MAE, RMSE and R2 for one set of predictions."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
    }


def kfold(seed: int = 42, n_splits: int = 5) -> KFold:
    return KFold(n_splits=n_splits, shuffle=True, random_state=seed)


def tune_and_score(pipeline, grid, X, y, seed: int = 42):
    """Fit GridSearchCV on X/y only; return fitted search and comparison table.

    The reported CV score is the mean validation RMSE over folds. For a grid
    with one value it equals ordinary 5-fold CV of that pipeline.
    """
    search = GridSearchCV(
        pipeline,
        grid,
        scoring="neg_root_mean_squared_error",
        cv=kfold(seed),
        n_jobs=-1,
        refit=True,
    )
    search.fit(X, y)
    results = search.cv_results_
    rows = []
    for index, params in enumerate(results["params"]):
        row = dict(params)
        row["cv_rmse"] = float(-results["mean_test_score"][index])
        row["cv_rmse_std"] = float(results["std_test_score"][index])
        rows.append(row)
    table = pd.DataFrame(rows)
    return search, table


def out_of_fold_predictions(best_estimator, X, y, seed: int = 42) -> np.ndarray:
    """Out-of-fold predictions used only for residual diagnostics."""
    return cross_val_predict(
        clone(best_estimator), X, y, cv=kfold(seed), n_jobs=-1
    )


def configuration_groups(X: pd.DataFrame) -> np.ndarray:
    """One group per physical configuration (angle, chord, U, delta*)."""
    keys = ["angle_deg", "chord_m", "velocity_m_s", "delta_star_m"]
    return X[keys].astype(str).agg("|".join, axis=1).to_numpy()


def grouped_cv_metrics(estimator, X, y_fit, groups, target_scale="log_pressure", n_splits: int = 5) -> dict:
    """Diagnostic CV that keeps all frequencies of one configuration together.

    Metrics are always returned in dB, even when the model was fitted on the
    logarithmic-pressure target SPL_dB / 20.
    """
    pred_fit = cross_val_predict(
        clone(estimator), X, y_fit, cv=GroupKFold(n_splits=n_splits), groups=groups, n_jobs=-1
    )
    return metrics(
        from_model_target(y_fit, target_scale),
        from_model_target(pred_fit, target_scale),
    )


def transfer_selection_table(validation: pd.DataFrame) -> pd.DataFrame:
    """Rank models by worst-case RMSE across held-out velocity regimes."""
    held = sorted(validation["held_out_U_m_s"].unique())
    rmse_wide = validation.pivot(index="model", columns="held_out_U_m_s", values="rmse")
    r2_wide = validation.pivot(index="model", columns="held_out_U_m_s", values="r2")

    rows = []
    for model in rmse_wide.index:
        rmse_values = rmse_wide.loc[model, held].to_numpy(dtype=float)
        r2_values = r2_wide.loc[model, held].to_numpy(dtype=float)
        row = {"model": model}
        for speed, value in zip(held, rmse_values):
            row[f"rmse_U{speed:g}"] = float(value)
        row["worst_rmse"] = float(np.max(rmse_values))
        row["mean_rmse"] = float(np.mean(rmse_values))
        row["worst_r2"] = float(np.min(r2_values))
        rows.append(row)
    return (
        pd.DataFrame(rows)
        .sort_values(["worst_rmse", "mean_rmse"])
        .reset_index(drop=True)
    )


def bootstrap_physical_coefficients(X, y, feature_set="physical", n_boot=400, seed=42):
    """Bootstrap coefficient distribution of an interpretable linear model.

    We bootstrap the unregularized linear model on the physical base set so
    that signs and magnitudes can be read directly. Regularized coefficients
    are not directly comparable across resamples because alpha would have to
    be re-tuned in every sample.
    """
    rng = np.random.default_rng(seed)
    columns = FEATURE_SETS[feature_set]
    pipeline = make_pipeline(feature_set, 1, LinearRegression())
    n = len(y)
    draws = []
    for _ in range(n_boot):
        index = rng.integers(0, n, size=n)
        pipeline.fit(X.iloc[index], np.asarray(y)[index])
        draws.append(pipeline.named_steps["model"].coef_)
    draws = np.asarray(draws)

    rows = []
    for j, name in enumerate(columns):
        values = draws[:, j]
        rows.append(
            {
                "feature": name,
                "median": float(np.median(values)),
                "q05": float(np.quantile(values, 0.05)),
                "q95": float(np.quantile(values, 0.95)),
                "positive_fraction": float(np.mean(values > 0)),
                "sd": float(np.std(values, ddof=1)),
            }
        )
    return pd.DataFrame(rows)


def residual_by_group(frame: pd.DataFrame, residuals, column: str, bins: int = 5):
    """Mean residual in quantile bins of one physical variable."""
    temp = pd.DataFrame({column: frame[column].to_numpy(), "residual": np.asarray(residuals)})
    temp["bin"] = pd.qcut(temp[column], q=bins, duplicates="drop")
    grouped = temp.groupby("bin", observed=True)["residual"]
    out = grouped.agg(count="count", mean="mean", median="median").reset_index()
    out[column + "_low"] = [interval.left for interval in out["bin"]]
    out[column + "_high"] = [interval.right for interval in out["bin"]]
    return out
