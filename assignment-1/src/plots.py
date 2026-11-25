"""Simple, readable diagnostic figures (matplotlib, Agg backend)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def measured_vs_predicted(y_true, y_pred, r2: float, path: Path) -> None:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    low = min(y_true.min(), y_pred.min()) - 2
    high = max(y_true.max(), y_pred.max()) + 2

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(y_true, y_pred, s=16, alpha=0.55, color="#286090")
    ax.plot([low, high], [low, high], "--", color="#444444", linewidth=1.2, label="ideal y = x")
    ax.set_xlim(low, high)
    ax.set_ylim(low, high)
    ax.set_xlabel("Measured SPL, dB")
    ax.set_ylabel("Predicted SPL, dB")
    ax.set_title(f"Measured vs predicted (test set), R2 = {r2:.3f}")
    ax.legend(loc="upper left")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def residual_diagnostics(frame, residuals, path: Path) -> None:
    residual = np.asarray(residuals, dtype=float)
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    panels = [
        ("predicted_spl", "Predicted SPL, dB"),
        ("velocity_m_s", "Velocity U, m/s"),
        ("log_fc_u", "log10(f*c/U)"),
        ("angle_deg", "Angle of attack, deg"),
    ]
    for ax, (column, label) in zip(axes.ravel(), panels):
        x = frame[column].to_numpy(dtype=float)
        ax.scatter(x, residual, s=14, alpha=0.5, color="#286090")
        ax.axhline(0.0, color="#444444", linewidth=1.0)
        ax.set_xlabel(label)
        ax.set_ylabel("Residual, dB")
        ax.grid(alpha=0.25)
    fig.suptitle("Test-set residuals: measured minus predicted", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def cv_summary_plot(table, path: Path) -> None:
    """CV RMSE for all candidates, sorted from best to worst."""
    plot_table = table.sort_values("cv_rmse")
    fig, ax = plt.subplots(figsize=(8.5, 5))
    y_pos = np.arange(len(plot_table))
    ax.barh(y_pos, plot_table["cv_rmse"], color="#7a9cc6")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(plot_table["model"])
    ax.invert_yaxis()
    ax.set_xlabel("5-fold CV RMSE on training data, dB")
    ax.set_title("Model comparison by cross-validation (lower is better)")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def regularization_plot(curves: dict, path: Path) -> None:
    """CV RMSE versus regularization strength for Ridge/Lasso/ElasticNet."""
    fig, ax = plt.subplots(figsize=(7, 5))
    for name, values in curves.items():
        ax.plot(values["x"], values["y"], marker="o", label=name)
    ax.set_xscale("log")
    ax.set_xlabel("Regularization strength alpha")
    ax.set_ylabel("5-fold CV RMSE, dB")
    ax.set_title("Validation error versus regularization strength")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
