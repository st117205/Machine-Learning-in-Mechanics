"""Data loading and feature construction.

All transformations are scikit-learn Pipeline steps. Therefore scaling,
polynomial expansion and feature selection are refitted inside every
cross-validation split; no information from a validation fold leaks into
training.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, PolynomialFeatures, StandardScaler

COLUMNS = [
    "frequency_hz",
    "angle_deg",
    "chord_m",
    "velocity_m_s",
    "delta_star_m",
    "spl_db",
]
TARGET = "spl_db"

# SPL_dB = 20 * log10(p / p_ref).  Working in the logarithmic pressure domain
# therefore means modelling SPL_dB / 20 and converting predictions back to dB.
DB_PER_LOG_PRESSURE = 20.0
TARGET_SCALE = "log_pressure"


def to_model_target(y, scale: str = TARGET_SCALE) -> np.ndarray:
    """Convert SPL in dB to the target scale used for fitting."""
    values = np.asarray(y, dtype=float)
    if scale == "db":
        return values
    if scale == "log_pressure":
        return values / DB_PER_LOG_PRESSURE
    raise ValueError(f"unknown target scale: {scale}")


def from_model_target(y_model, scale: str = TARGET_SCALE) -> np.ndarray:
    """Convert fitted target values back to SPL in dB."""
    values = np.asarray(y_model, dtype=float)
    if scale == "db":
        return values
    if scale == "log_pressure":
        return values * DB_PER_LOG_PRESSURE
    raise ValueError(f"unknown target scale: {scale}")


# Physically motivated base set: logarithms of frequency, velocity, chord and
# boundary-layer thickness plus the geometrical angle of attack.
PHYSICAL = ["log_f", "log_u", "log_delta", "angle_deg", "log_c"]

FEATURE_SETS = {
    "raw": ["frequency_hz", "angle_deg", "chord_m", "velocity_m_s", "delta_star_m"],
    "physical": PHYSICAL,
    # Strouhal-style variant: f*c/U replaces the separate log_f and log_u.
    "strouhal": ["log_fc_u", "log_delta", "angle_deg", "log_c"],
    # Same physical base, with an explicit U^2 term to test the "U powers" idea.
    "physical_u2": ["log_f", "log_u", "u2", "log_delta", "angle_deg", "log_c"],
}


def load_data(path: Path | str) -> pd.DataFrame:
    """Read the UCI .dat file (tab separated, no header)."""
    return pd.read_csv(path, sep=r"\s+", header=None, names=COLUMNS, engine="python")


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived, physically motivated features. No target information used."""
    out = df.copy()
    out["log_f"] = np.log10(out["frequency_hz"])
    out["log_u"] = np.log10(out["velocity_m_s"])
    out["log_c"] = np.log10(out["chord_m"])
    out["log_delta"] = np.log10(out["delta_star_m"])
    out["fc_u"] = out["frequency_hz"] * out["chord_m"] / out["velocity_m_s"]
    out["log_fc_u"] = np.log10(out["fc_u"])
    out["u2"] = out["velocity_m_s"] ** 2
    return out


def _select_columns(frame, columns):
    """Select columns inside a Pipeline step."""
    return np.asarray(frame[columns], dtype=float)


def make_pipeline(feature_set: str, degree: int, estimator):
    """Build a Pipeline: select -> (poly) -> standardize -> estimator."""
    columns = FEATURE_SETS[feature_set]
    steps = [
        ("select", FunctionTransformer(_select_columns, kw_args={"columns": columns}, validate=False)),
    ]
    if degree > 1:
        steps.append(("poly", PolynomialFeatures(degree=degree, include_bias=False)))
    steps.append(("scale", StandardScaler()))
    steps.append(("model", estimator))
    return Pipeline(steps)


def make_baseline() -> Pipeline:
    """Dummy model: always predicts the training mean."""
    return Pipeline([("model", DummyRegressor(strategy="mean"))])


def make_estimator(kind: str):
    """Return a fresh estimator for the requested regularized model."""
    if kind == "ols":
        return LinearRegression()
    if kind == "ridge":
        return Ridge(random_state=0)
    if kind == "lasso":
        return Lasso(max_iter=100000, tol=1e-3, random_state=0)
    if kind == "elasticnet":
        return ElasticNet(max_iter=100000, tol=1e-3, random_state=0)
    raise ValueError(f"unknown estimator kind: {kind}")


def model_grid(name: str):
    """Return (pipeline, parameter grid) for one candidate model.

    Every grid is evaluated on the training part only. GridSearchCV refits
    the whole pipeline in each fold, so standardization and polynomial
    features are always computed from that fold's training data.
    """
    if name == "baseline":
        return make_baseline(), [{}]
    if name == "ols_raw":
        return make_pipeline("raw", 1, make_estimator("ols")), [{}]
    if name == "ols_physical":
        return make_pipeline("physical", 1, make_estimator("ols")), [{}]
    if name == "ols_strouhal":
        return make_pipeline("strouhal", 1, make_estimator("ols")), [{}]
    if name == "ols_physical_u2":
        return make_pipeline("physical_u2", 1, make_estimator("ols")), [{}]
    if name == "poly1_physical":
        return make_pipeline("physical", 1, make_estimator("ols")), [{}]
    if name == "poly2_physical":
        return make_pipeline("physical", 2, make_estimator("ols")), [{}]
    if name == "poly3_physical":
        return make_pipeline("physical", 3, make_estimator("ols")), [{}]
    if name == "ridge_poly2":
        return make_pipeline("physical", 2, make_estimator("ridge")), {
            "model__alpha": [0.01, 0.1, 1.0, 10.0, 100.0]
        }
    if name == "lasso_poly2":
        return make_pipeline("physical", 2, make_estimator("lasso")), {
            "model__alpha": [0.001, 0.01, 0.1, 1.0, 10.0]
        }
    if name == "elasticnet_poly2":
        return make_pipeline("physical", 2, make_estimator("elasticnet")), {
            "model__alpha": [0.01, 0.1, 1.0, 10.0],
            "model__l1_ratio": [0.2, 0.5, 0.8],
        }
    raise ValueError(f"unknown model name: {name}")


MODEL_ORDER = [
    "baseline",
    "ols_raw",
    "ols_physical",
    "ols_strouhal",
    "ols_physical_u2",
    "poly2_physical",
    "poly3_physical",
    "ridge_poly2",
    "lasso_poly2",
    "elasticnet_poly2",
]
