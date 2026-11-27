"""Reproduce the whole airfoil self-noise analysis.

Run from assignment-1:  py -m src.main

Important design choices:
- The target is modelled in the acoustic log-pressure domain SPL_dB / 20 and
  converted back to dB for every reported metric.
- The final model is NOT chosen by the ordinary random CV. It is chosen by
  worst-case RMSE on two fully held-out velocity regimes (55.5 and 71.3 m/s),
  i.e. by transfer to unfamiliar regimes. The random test set is only used
  after that choice.
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
    TARGET_SCALE,
    add_features,
    from_model_target,
    load_data,
    model_grid,
    to_model_target,
)
from .evaluate import (
    bootstrap_physical_coefficients,
    configuration_groups,
    grouped_cv_metrics,
    metrics,
    out_of_fold_predictions,
    residual_by_group,
    transfer_selection_table,
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
HELD_OUT_VELOCITIES = (55.5, 71.3)
SELECTION_CRITERION = (
    "Choose the model with the smallest worst-case RMSE on the two held-out "
    "velocity regimes; break ties by mean RMSE. This measures transfer to "
    "unfamiliar regimes instead of random interpolation."
)


def score_to_db(scores, scale: str = TARGET_SCALE) -> np.ndarray:
    """Convert RMSE-like values from the fitted target scale back to dB."""
    return np.atleast_1d(from_model_target(scores, scale))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "selection_criterion.txt").write_text(SELECTION_CRITERION, encoding="utf-8")

    df = add_features(load_data(DATA_PATH))
    y_db = df[TARGET].to_numpy(dtype=float)
    y_fit = to_model_target(y_db, TARGET_SCALE)
    print(
        f"Loaded {len(df)} rows; fitting target = {TARGET_SCALE} "
        f"(SPL_dB / 20), metrics reported in dB"
    )

    # ---------- Ordinary interpolation split, kept untouched for final test ----------
    train_idx, test_idx = train_test_split(
        np.arange(len(df)),
        test_size=0.20,
        random_state=42,
        stratify=df["velocity_m_s"],
    )
    X_train, X_test = df.iloc[train_idx].copy(), df.iloc[test_idx].copy()
    y_train_db, y_test_db = y_db[train_idx], y_db[test_idx]
    y_train_fit = y_fit[train_idx]
    print(f"Random split: train={len(y_train_db)}, test={len(y_test_db)}")

    # ---------- Experiment A: ordinary CV, diagnostics only ----------
    rows = []
    cv_tables = {}
    fitted = {}
    for name in MODEL_ORDER:
        pipeline, grid = model_grid(name)
        search, table = tune_and_score(pipeline, grid, X_train, y_train_fit)
        best = search.best_estimator_

        train_pred_db = from_model_target(best.predict(X_train), TARGET_SCALE)
        test_pred_db = from_model_target(best.predict(X_test), TARGET_SCALE)
        train_metrics = metrics(y_train_db, train_pred_db)
        test_metrics = metrics(y_test_db, test_pred_db)

        rows.append(
            {
                "model": name,
                "cv_rmse": float(score_to_db(-search.best_score_)[0]),
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
            f"{name:20s} CV={float(score_to_db(-search.best_score_)[0]):6.3f} dB  "
            f"train={train_metrics['rmse']:6.3f} dB  test={test_metrics['rmse']:6.3f} dB"
        )

    summary = pd.DataFrame(rows).sort_values("cv_rmse").reset_index(drop=True)
    cv_best = summary.loc[0, "model"]
    print(f"\nBest by ordinary random CV (diagnostic only): {cv_best}")

    # ---------- Experiment B: transfer validation, this chooses the final model ----------
    hold_rows = []
    for held_u in HELD_OUT_VELOCITIES:
        mask_train = df["velocity_m_s"].to_numpy() != held_u
        Xh_train, Xh_test = df.iloc[mask_train].copy(), df.iloc[~mask_train].copy()
        yh_train_fit = y_fit[mask_train]
        yh_test_db = y_db[~mask_train]

        for name in MODEL_ORDER:
            pipeline, grid = model_grid(name)
            search_h, _ = tune_and_score(pipeline, grid, Xh_train, yh_train_fit)
            pred_db = from_model_target(
                search_h.best_estimator_.predict(Xh_test), TARGET_SCALE
            )
            result = metrics(yh_test_db, pred_db)

            hold_frame = Xh_test.copy()
            hold_frame["measured_spl"] = yh_test_db
            hold_frame["predicted_spl"] = pred_db
            hold_frame["residual"] = yh_test_db - pred_db
            hold_frame.to_csv(OUT / f"holdout_predictions_U{held_u:g}_{name}.csv", index=False)

            hold_rows.append(
                {
                    "held_out_U_m_s": held_u,
                    "model": name,
                    "n_test": len(yh_test_db),
                    "cv_rmse_train": float(score_to_db(-search_h.best_score_)[0]),
                    "best_params": str(search_h.best_params_),
                    "mae": result["mae"],
                    "rmse": result["rmse"],
                    "r2": result["r2"],
                }
            )
        current = [row for row in hold_rows if row["held_out_U_m_s"] == held_u]
        text = ", ".join(f"{row['model']}={row['rmse']:.3f} dB" for row in current)
        print(f"Held-out U={held_u:g}: {text}")

    transfer = pd.DataFrame(hold_rows)
    transfer.to_csv(OUT / "transfer_validation.csv", index=False)

    selection = transfer_selection_table(transfer)
    selection = selection.merge(summary[["model", "cv_rmse"]], on="model", how="left")
    selection.to_csv(OUT / "model_selection.csv", index=False)
    selected = selection.loc[0, "model"]
    print("\nSelection by worst-case held-out RMSE:")
    print(selection.head(5).to_string(index=False))
    print(f"\nFINAL MODEL: {selected}")

    # Out-of-fold predictions of the final model, diagnostics only.
    oof_fit = out_of_fold_predictions(fitted[selected], X_train, y_train_fit)
    oof_db = from_model_target(oof_fit, TARGET_SCALE)
    oof_metrics = metrics(y_train_db, oof_db)
    summary.loc[summary["model"] == selected, ["cv_mae", "cv_rmse_oof", "cv_r2"]] = [
        oof_metrics["mae"], oof_metrics["rmse"], oof_metrics["r2"]
    ]
    summary.to_csv(OUT / "model_comparison.csv", index=False)

    final_model = clone(fitted[selected]).fit(X_train, y_train_fit)
    test_pred_db = from_model_target(final_model.predict(X_test), TARGET_SCALE)
    residual = y_test_db - test_pred_db
    pred_frame = X_test.copy()
    pred_frame["measured_spl"] = y_test_db
    pred_frame["predicted_spl"] = test_pred_db
    pred_frame["residual"] = residual
    pred_frame.to_csv(OUT / "predictions_test.csv", index=False)

    test_metrics = metrics(y_test_db, test_pred_db)
    print(
        f"Final model on random test: MAE={test_metrics['mae']:.3f} dB, "
        f"RMSE={test_metrics['rmse']:.3f} dB, R2={test_metrics['r2']:.3f}"
    )

    # ---------- Target-scale check: dB versus acoustic log-pressure ----------
    scale_rows = []
    for scale in ["db", "log_pressure"]:
        y_train_scale = to_model_target(y_train_db, scale)
        pipeline_s, grid_s = model_grid(selected)
        search_s, _ = tune_and_score(pipeline_s, grid_s, X_train, y_train_scale)
        pred_db = from_model_target(search_s.best_estimator_.predict(X_test), scale)
        result = metrics(y_test_db, pred_db)
        scale_rows.append(
            {
                "target_scale": scale,
                "best_params": str(search_s.best_params_),
                "test_mae_db": result["mae"],
                "test_rmse_db": result["rmse"],
                "test_r2": result["r2"],
            }
        )
    scale_check = pd.DataFrame(scale_rows)
    scale_check.to_csv(OUT / "target_scale_check.csv", index=False)
    print("\nTarget-scale check (both converted back to dB):")
    print(scale_check.to_string(index=False))

    # ---------- Figures and diagnostic tables ----------
    measured_vs_predicted(y_test_db, test_pred_db, test_metrics["r2"], OUT / "measured_vs_predicted.png")
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
        table["cv_rmse_fit"] = table["cv_rmse"]
        table["cv_rmse"] = score_to_db(table["cv_rmse"], TARGET_SCALE)
        table.to_csv(OUT / f"cv_{name}.csv", index=False)
        if name == "elasticnet_poly2":
            table = table[np.isclose(table["model__l1_ratio"], 0.5)]
        curves[name] = {"x": table["model__alpha"].to_numpy(), "y": table["cv_rmse"].to_numpy()}
    regularization_plot(curves, OUT / "cv_regularization.png")

    # Bootstrap stays in dB so coefficients remain directly interpretable.
    boot = bootstrap_physical_coefficients(X_train, y_train_db, "physical", n_boot=400)
    boot.to_csv(OUT / "bootstrap_coefficients.csv", index=False)
    boot_st = bootstrap_physical_coefficients(X_train, y_train_db, "strouhal", n_boot=400)
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
        eligible = grouped[grouped["count"] >= 10]
        worst = eligible.iloc[eligible["mean"].abs().argmax()]
        print(
            f"Largest mean residual by {column}: "
            f"{worst['mean']:+.2f} dB in [{worst[column + '_low']:.3g}, {worst[column + '_high']:.3g}], "
            f"n={int(worst['count'])}"
        )

    # Extra leakage check, diagnostics only; never used for selection.
    groups_train = configuration_groups(X_train)
    group_rows = []
    for name in dict.fromkeys([selected, cv_best]):
        result = grouped_cv_metrics(
            fitted[name], X_train, y_train_fit, groups_train, target_scale=TARGET_SCALE
        )
        group_rows.append({"model": name, "n_groups": len(np.unique(groups_train)), **result})
    grouped_summary = pd.DataFrame(group_rows)
    grouped_summary.to_csv(OUT / "grouped_cv_metrics.csv", index=False)
    print("\nGrouped-CV diagnostic (whole configuration held out):")
    for _, row in grouped_summary.iterrows():
        print(
            f"  {row['model']:18s} RMSE={row['rmse']:.3f} dB, "
            f"MAE={row['mae']:.3f} dB, R2={row['r2']:.3f}"
        )

    print(f"\nAll tables and figures written to {OUT}")


if __name__ == "__main__":
    main()
