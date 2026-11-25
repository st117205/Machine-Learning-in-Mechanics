"""Unit tests for the pure calculation layer."""

import unittest

from poisson_lab.calculations import (
    axial_strain,
    estimate_shear_modulus_gpa,
    is_stable_poisson,
    percent_difference,
    poisson_from_measurement,
    poisson_from_readings,
    transverse_strain,
    within_reference_range,
)
from poisson_lab.materials import MATERIAL_DATABASE


class StrainTests(unittest.TestCase):
    def test_axial_strain(self):
        self.assertAlmostEqual(axial_strain(0.1, 50.0), 0.002)

    def test_transverse_strain(self):
        self.assertAlmostEqual(transverse_strain(0.006, 10.0), 0.0006)

    def test_negative_contraction_is_allowed(self):
        self.assertAlmostEqual(transverse_strain(-0.006, 10.0), -0.0006)

    def test_zero_gauge_length_is_rejected(self):
        with self.assertRaises(ValueError):
            axial_strain(0.1, 0.0)

    def test_nan_is_rejected(self):
        with self.assertRaises(ValueError):
            axial_strain(float("nan"), 50.0)


class PoissonTests(unittest.TestCase):
    def test_single_measurement(self):
        nu = poisson_from_measurement(50.0, 10.0, 0.10, 0.006)
        self.assertAlmostEqual(nu, 0.30)

    def test_multiple_readings_use_ratio_of_mean_strains(self):
        readings = [
            {"extension_mm": 0.10, "contraction_mm": 0.006},
            {"extension_mm": 0.20, "contraction_mm": 0.012},
        ]
        self.assertAlmostEqual(poisson_from_readings(50.0, 10.0, readings), 0.30)

    def test_zero_extension_is_rejected(self):
        with self.assertRaises(ValueError):
            poisson_from_measurement(50.0, 10.0, 0.0, 0.006)

    def test_empty_readings_is_rejected(self):
        with self.assertRaises(ValueError):
            poisson_from_readings(50.0, 10.0, [])


class ReferenceTests(unittest.TestCase):
    def test_database_has_all_13_original_materials(self):
        self.assertEqual(len(MATERIAL_DATABASE), 13)
        expected = {
            "iron", "steel", "stainless_steel", "aluminum", "copper", "brass",
            "oak_wood", "pine_wood", "teak_wood", "bamboo", "plywood", "pvc", "acrylic",
        }
        self.assertEqual(set(MATERIAL_DATABASE), expected)

    def test_percent_difference(self):
        self.assertAlmostEqual(percent_difference(0.28, 0.30), 6.6667, places=3)

    def test_within_reference_range(self):
        self.assertTrue(within_reference_range(0.30, "steel"))
        self.assertFalse(within_reference_range(0.50, "steel"))

    def test_stability_range(self):
        self.assertTrue(is_stable_poisson(0.30))
        self.assertFalse(is_stable_poisson(0.50))
        self.assertFalse(is_stable_poisson(-1.0))

    def test_shear_modulus_estimate(self):
        self.assertAlmostEqual(estimate_shear_modulus_gpa(200.0, 0.30), 76.9231, places=3)

    def test_shear_modulus_rejects_invalid_nu(self):
        with self.assertRaises(ValueError):
            estimate_shear_modulus_gpa(200.0, 0.5)


if __name__ == "__main__":
    unittest.main()
