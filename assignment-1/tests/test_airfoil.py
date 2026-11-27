"""Small sanity checks for the analysis code."""

import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.airfoil import (
    DB_PER_LOG_PRESSURE,
    FEATURE_SETS,
    MODEL_ORDER,
    add_features,
    from_model_target,
    load_data,
    model_grid,
    to_model_target,
)
from src.evaluate import metrics, transfer_selection_table


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "airfoil_self_noise.dat"


class DataTests(unittest.TestCase):
    def test_dataset_shape(self):
        df = load_data(DATA_PATH)
        self.assertEqual(df.shape, (1503, 6))
        self.assertFalse(df.isna().any().any())
        self.assertEqual(len(df.drop_duplicates()), len(df))

    def test_derived_features_are_finite(self):
        df = add_features(load_data(DATA_PATH))
        for name in ["log_f", "log_u", "log_c", "log_delta", "log_fc_u", "u2"]:
            self.assertTrue(np.isfinite(df[name].to_numpy()).all())

    def test_strouhal_identity(self):
        df = add_features(load_data(DATA_PATH))
        expected = np.log10(df["frequency_hz"] * df["chord_m"] / df["velocity_m_s"])
        np.testing.assert_allclose(df["log_fc_u"].to_numpy(), expected.to_numpy())


class ModelTests(unittest.TestCase):
    def test_all_feature_sets_have_columns(self):
        df = add_features(load_data(DATA_PATH))
        for columns in FEATURE_SETS.values():
            for name in columns:
                self.assertIn(name, df.columns)

    def test_every_model_has_a_grid(self):
        for name in MODEL_ORDER:
            pipeline, grid = model_grid(name)
            self.assertIsNotNone(pipeline)
            self.assertTrue(len(grid) >= 1)


class MetricTests(unittest.TestCase):
    def test_perfect_prediction(self):
        y = np.array([1.0, 2.0, 3.0])
        result = metrics(y, y)
        self.assertAlmostEqual(result["mae"], 0.0)
        self.assertAlmostEqual(result["rmse"], 0.0)
        self.assertAlmostEqual(result["r2"], 1.0)

    def test_known_mae(self):
        result = metrics(np.array([1.0, 2.0]), np.array([2.0, 4.0]))
        self.assertAlmostEqual(result["mae"], 1.5)
        self.assertAlmostEqual(result["rmse"], np.sqrt(2.5))


class TargetScaleTests(unittest.TestCase):
    def test_db_scale_is_identity(self):
        y = np.array([100.0, 124.836, 141.0])
        np.testing.assert_allclose(to_model_target(y, "db"), y)
        np.testing.assert_allclose(from_model_target(y, "db"), y)

    def test_log_pressure_definition(self):
        y = np.array([100.0, 124.836, 141.0])
        expected = y / DB_PER_LOG_PRESSURE
        np.testing.assert_allclose(to_model_target(y, "log_pressure"), expected)

    def test_log_pressure_roundtrip(self):
        y = np.array([103.38, 124.836, 140.987])
        back = from_model_target(to_model_target(y, "log_pressure"), "log_pressure")
        np.testing.assert_allclose(back, y)


class SelectionTests(unittest.TestCase):
    def test_worst_case_selection(self):
        frame = pd.DataFrame([
            {"model": "poly3", "held_out_U_m_s": 55.5, "rmse": 2.0, "r2": 0.9},
            {"model": "poly3", "held_out_U_m_s": 71.3, "rmse": 5.0, "r2": 0.5},
            {"model": "poly2", "held_out_U_m_s": 55.5, "rmse": 3.0, "r2": 0.8},
            {"model": "poly2", "held_out_U_m_s": 71.3, "rmse": 3.5, "r2": 0.76},
        ])
        result = transfer_selection_table(frame)
        self.assertEqual(result.loc[0, "model"], "poly2")
        self.assertAlmostEqual(result.loc[0, "worst_rmse"], 3.5)
        self.assertAlmostEqual(result.loc[0, "mean_rmse"], 3.25)
        self.assertAlmostEqual(result.loc[0, "worst_r2"], 0.76)


if __name__ == "__main__":
    unittest.main()
