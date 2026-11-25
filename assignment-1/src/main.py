"""Reproduce the whole airfoil self-noise analysis.

Run from assignment-1:  py -m src.main
All tunable choices are made by cross-validation on training data only.
The random test set and the held-out velocity sets are never used for tuning.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import train_test_split

from .airfoil import (
    FEATURE_SETS,
    MODEL_ORDER,
    TARGET,
    add_features,
    load_data,
    model_grid,
)
from .evaluate import (
    bootstrap_physical_coefficients,
    configuration_groups,
    grouped_cv_metrics,
    metrics,
    out_of_fold_predictions,
    residual_by_group,
    tune_and_score,
)
from .plots import (
    cv_summary_plot,
    measured_vs_predicted,
    regularization_plot,
    residual_diagnostics,
)

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "airfoil_self_noise.dat"
OUT = ROOT / "results"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    df = add_features(load_data(DATA_PATH))
    y = df[TARGET].to_numpy(dtype=float)
    print(f"Loaded {len(df)} rows, {len(df.columns)} columns (including derived features)")

    # ---------- Experiment A: ordinary interpolation test ----------
    train_idx, test_idx = train_test_split(
        np.arange(len(df)),
        test_size=0.20,
        random_state=42,
        stratify=df["velocity_m_s"],
    )
    X_train, X_test = df.iloc[train_idx].copy(), df.iloc[test_idx].copy()
    y_train, y_test = y[train_idx], y[test_idx]
    print(f"Random split: train={len(y_train)}, test={len(y_test)}")

    rows = []
    cv_tables = {}
    fitted = {}
    for name in MODEL_ORDER:
        pipeline, grid = model_grid(name)
        search, table = tune_and_score(pipeline, grid, X_train, y_train)
        best = search.best_estimator_
        train_metrics = metrics(y_train, best.predict(X_train))
        test_metrics = metrics(y_test, best.predict(X_test))
        rows.append(
            {
                "model": name,
                "cv_rmse": float(-search.best_score_),
                "best_params": str(search.best_params_),
                "train_mae": train_metrics["mae"],
                "train_rmse": train_metrics["rmse"],
                "train_r2": train_metrics["r2"],
                "cv_mae": np.nan,
                "cv_rmse_oof": np.nan,
                "cv_r2": np.nan,
                "test_mae": test_metrics["mae"],
                "test_rmse": test_metrics["rmse"],
                "test_r2": test_metrics["r2"],
            }
        )
        cv_tables[name] = table
        fitted[name] = best
        print(
            f"{name:20s} cv_rmse={-search.best_score_:6.3f}  "
            f"train_rmse={train_metrics['rmse']:6.3f}  test_rmse={test_metrics['rmse']:6.3f}"
        )

    summary = pd.DataFrame(rows).sort_values("cv_rmse").reset_index(drop=True)
    selected = summary.loc[0, "model"]
    print(f"\nSelected by training CV: {selected} ({summary.loc[0, 'best_params']})")

    # Out-of-fold predictions of the selected model, used for diagnostics only.
    oof = out_of_fold_predictions(fitted[selected], X_train, y_train)
    oof_metrics = metrics(y_train, oof)
    summary.loc[summary["model"] == selected, ["cv_mae", "cv_rmse_oof", "cv_r2"]] = [
        oof_metrics["mae"], oof_metrics["rmse"], oof_metrics["r2"]
    ]
    summary.to_csv(OUT / "model_comparison.csv", index=False)

    # Extra leakage check only: keep every measured frequency of one physical
    # configuration in the same fold. This is diagnostic, not used for tuning.
    groups_train = configuration_groups(X_train)
    group_rows = []
    for name in dict.fromkeys([selected, "poly2_physical"]):
        result = grouped_cv_metrics(fitted[name], X_train, y_train, groups_train)
        group_rows.append({"model": name, "n_groups": len(np.unique(groups_train)), **result})
    grouped_summary = pd.DataFrame(group_rows)
    grouped_summary.to_csv(OUT / "grouped_cv_metrics.csv", index=False)
    print("\nGrouped-CV diagnostic (whole configuration held out):")
    for _, row in grouped_summary.iterrows():
        print(
            f"  {row['model']:18s} RMSE={row['rmse']:.3f} dB, "
            f"MAE={row['mae']:.3f} dB, R2={row['r2']:.3f}"
        )

    final_model = clone(fitted[selected]).fit(X_train, y_train)
    test_pred = final_model.predict(X_test)
    residual = y_test - test_pred
    pred_frame = X_test.copy()
    pred_frame["measured_spl"] = y_test
    pred_frame["predicted_spl"] = test_pred
    pred_frame["residual"] = residual
    pred_frame.to_csv(OUT / "predictions_test.csv", index=False)

    test_metrics = metrics(y_test, test_pred)
    print(
        f"Selected model on random test: MAE={test_metrics['mae']:.3f} dB, "
        f"RMSE={test_metrics['rmse']:.3f} dB, R2={test_metrics['r2']:.3f}"
    )

    # ---------- Figures and diagnostic tables ----------
    measured_vs_predicted(y_test, test_pred, test_metrics["r2"], OUT / "measured_vs_predicted.png")
    residual_diagnostics(pred_frame, residual, OUT / "residual_diagnostics.png")
    cv_summary_plot(summary, OUT / "cv_model_comparison.png")

    degree_rows = []
    for name, degree in [("ols_physical", 1), ("poly2_physical", 2), ("poly3_physical", 3)]:
        row = summary[summary["model"] == name].iloc[0]
        degree_rows.append(
            {
                "degree": degree,
                "cv_rmse": row["cv_rmse"],
                "train_rmse": row["train_rmse"],
                "test_rmse": row["test_rmse"],
            }
        )
    pd.DataFrame(degree_rows).to_csv(OUT / "cv_degree.csv", index=False)

    curves = {}
    for name in ["ridge_poly2", "lasso_poly2", "elasticnet_poly2"]:
        table = cv_tables[name].copy()
        table.to_csv(OUT / f"cv_{name}.csv", index=False)
        if name == "elasticnet_poly2":
            table = table[np.isclose(table["model__l1_ratio"], 0.5)]
        curves[name] = {"x": table["model__alpha"].to_numpy(), "y": table["cv_rmse"].to_numpy()}
    regularization_plot(curves, OUT / "cv_regularization.png")

    boot = bootstrap_physical_coefficients(X_train, y_train, "physical", n_boot=400)
    boot.to_csv(OUT / "bootstrap_coefficients.csv", index=False)
    boot_st = bootstrap_physical_coefficients(X_train, y_train, "strouhal", n_boot=400)
    boot_st.to_csv(OUT / "bootstrap_coefficients_strouhal.csv", index=False)
    print("\nBootstrap of physical linear coefficients (median [q05, q95], P(sign>0)):")
    for _, row in boot.iterrows():
        print(
            f"  {row['feature']:12s} {row['median']:+7.3f} "
            f"[{row['q05']:+7.3f}, {row['q95']:+7.3f}]  P(+)={row['positive_fraction']:.2f}"
        )

    corr = X_train[FEATURE_SETS["physical"] + ["log_fc_u"]].corr()
    corr.to_csv(OUT / "feature_correlations.csv")

    for column in ["velocity_m_s", "angle_deg", "log_fc_u", "predicted_spl"]:
        grouped = residual_by_group(pred_frame, residual, column, bins=5)
        grouped.to_csv(OUT / f"residual_by_{column}.csv", index=False)
        worst = grouped[grouped["count"] >= 10].iloc[grouped[grouped["count"] >= 10]["mean"].abs().argmax()]
        print(
            f"Largest mean residual by {column}: "
            f"{worst['mean']:+.2f} dB in [{worst[column + '_low']:.3g}, {worst[column + '_high']:.3g}], "
            f"n={int(worst['count'])}"
        )

    # ---------- Experiment B: held-out physical regimes ----------
    hold_rows = []
    comparison_models = [name for name in dict.fromkeys([selected, "poly2_physical", "baseline"])]
    for held_u in [55.5, 71.3]:
        mask_train = df["velocity_m_s"] != held_u
        Xh_train, Xh_test = df[mask_train].copy(), df[~mask_train].copy()
        yh_train, yh_test = y[mask_train.to_numpy()], y[~mask_train.to_numpy()]
        for name in comparison_models:
            pipeline, grid = model_grid(name)
            search_h, _ = tune_and_score(pipeline, grid, Xh_train, yh_train)
            pred_h = search_h.best_estimator_.predict(Xh_test)
            result = metrics(yh_test, pred_h)
            hold_frame = Xh_test.copy()
            hold_frame["measured_spl"] = yh_test
            hold_frame["predicted_spl"] = pred_h
            hold_frame["residual"] = yh_test - pred_h
            hold_frame.to_csv(OUT / f"holdout_predictions_U{held_u}_{name}.csv", index=False)
            hold_rows.append(
                {
                    "held_out_U_m_s": held_u,
                    "model": name,
                    "n_test": len(yh_test),
                    "cv_rmse_train": float(-search_h.best_score_),
                    "best_params": str(search_h.best_params_),
                    "mae": result["mae"],
                    "rmse": result["rmse"],
                    "r2": result["r2"],
                }
            )
        current = [row for row in hold_rows if row["held_out_U_m_s"] == held_u]
        summary_text = ", ".join(f"{row['model']}={row['rmse']:.3f} dB" for row in current)
        print(f"Held-out U={held_u}: n={len(yh_test)}, RMSE: {summary_text}")
        for row in current:
            frame = pd.read_csv(OUT / f"holdout_predictions_U{held_u}_{row['model']}.csv")
            print(f"  mean residual ({row['model']}): {frame['residual'].mean():+.3f} dB")
    pd.DataFrame(hold_rows).to_csv(OUT / "holdout_metrics.csv", index=False)
    print(f"\nAll tables and figures written to {OUT}")


if __name__ == "__main__":
    main()
