"""Pure calculation functions for the Poisson-ratio laboratory.

Sign convention used everywhere:
  * axial_extension_mm  > 0  when the specimen gets longer (tension);
  * transverse_contraction_mm > 0 when the specimen gets thinner.

With this convention a normal material gives a positive Poisson ratio:

    nu = (transverse strain) / (axial strain)
       = (contraction / width) / (extension / gauge_length)

All lengths are entered in millimetres. Strains are dimensionless, so no
unit conversion is needed as long as the same unit is used for numerator and
denominator.
"""

from __future__ import annotations

import math
from typing import Dict, Iterable, List

from .materials import MATERIAL_DATABASE


def _finite_number(value, name: str) -> float:
    """Convert to float and reject NaN/infinity."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a number") from None
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def axial_strain(extension_mm: float, gauge_length_mm: float) -> float:
    """Return epsilon_axial = extension / gauge_length."""
    gauge = _finite_number(gauge_length_mm, "gauge_length_mm")
    if gauge <= 0:
        raise ValueError("gauge_length_mm must be positive")
    return _finite_number(extension_mm, "extension_mm") / gauge


def transverse_strain(contraction_mm: float, width_mm: float) -> float:
    """Return epsilon_transverse = contraction / original width."""
    width = _finite_number(width_mm, "width_mm")
    if width <= 0:
        raise ValueError("width_mm must be positive")
    return _finite_number(contraction_mm, "contraction_mm") / width


def poisson_from_measurement(
    gauge_length_mm: float,
    width_mm: float,
    extension_mm: float,
    contraction_mm: float,
) -> float:
    """Compute nu from one pair of axial and transverse measurements."""
    axial = axial_strain(extension_mm, gauge_length_mm)
    if axial == 0:
        raise ValueError("extension_mm must be non-zero")
    return transverse_strain(contraction_mm, width_mm) / axial


def poisson_from_readings(
    gauge_length_mm: float,
    width_mm: float,
    readings: Iterable[Dict[str, float]],
) -> float:
    """Compute nu from several readings as the ratio of the mean strains.

    Using the ratio of mean strains is equivalent to a least-squares fit
    through the origin and is more stable than averaging individual ratios.
    """
    axial_values: List[float] = []
    transverse_values: List[float] = []

    for index, reading in enumerate(readings, 1):
        extension = reading["extension_mm"]
        contraction = reading["contraction_mm"]
        if _finite_number(extension, f"reading {index} extension") == 0:
            raise ValueError(f"reading {index}: extension_mm must be non-zero")
        axial_values.append(axial_strain(extension, gauge_length_mm))
        transverse_values.append(transverse_strain(contraction, width_mm))

    if not axial_values:
        raise ValueError("at least one reading is required")

    mean_axial = sum(axial_values) / len(axial_values)
    return sum(transverse_values) / len(transverse_values) / mean_axial


def percent_difference(measured: float, reference: float) -> float:
    """Return absolute percentage difference from the reference value."""
    ref = _finite_number(reference, "reference")
    if ref == 0:
        raise ValueError("reference must be non-zero")
    return abs(_finite_number(measured, "measured") - ref) / abs(ref) * 100.0


def is_stable_poisson(nu: float) -> bool:
    """Check the isotropic-elasticity stability range -1 < nu < 0.5."""
    return -1.0 < _finite_number(nu, "nu") < 0.5


def within_reference_range(nu: float, material_key: str) -> bool:
    """Check whether nu lies in the material's teaching reference range."""
    material = MATERIAL_DATABASE[material_key]
    value = _finite_number(nu, "nu")
    return material["poisson_min"] <= value <= material["poisson_max"]


def estimate_shear_modulus_gpa(youngs_modulus_gpa: float, nu: float) -> float:
    """Estimate G from E = 2G(1 + nu); only valid for isotropic materials."""
    e = _finite_number(youngs_modulus_gpa, "youngs_modulus_gpa")
    if not is_stable_poisson(nu):
        raise ValueError("nu must satisfy -1 < nu < 0.5")
    return e / (2.0 * (1.0 + _finite_number(nu, "nu")))
