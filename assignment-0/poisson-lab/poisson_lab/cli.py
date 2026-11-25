"""Command-line interface for the Poisson-ratio laboratory.

The output is intentionally ASCII-only so it works on Windows consoles with
non-UTF-8 code pages (the original young-modulus project crashed on 'cm^3'
characters in such environments).
"""

from __future__ import annotations

import argparse
from typing import Dict, List, Optional, Sequence

from .calculations import (
    axial_strain,
    estimate_shear_modulus_gpa,
    is_stable_poisson,
    percent_difference,
    poisson_from_measurement,
    poisson_from_readings,
    transverse_strain,
    within_reference_range,
)
from .materials import MATERIAL_DATABASE

LINE = "=" * 76
DASH = "-" * 76


def print_material_table() -> None:
    """Print all 13 materials with reference values and a derived G estimate."""
    print(LINE)
    print("REFERENCE MATERIALS (same 13 materials as the young-modulus project)")
    print(LINE)
    print(f"{'#':<3} {'Key':<18} {'Name':<18} {'E (GPa)':>8} {'nu ref':>7} "
          f"{'range':>13} {'G est (GPa)':>12}")
    print(DASH)
    for index, (key, material) in enumerate(MATERIAL_DATABASE.items(), 1):
        if material["isotropic"]:
            g_est = f"{estimate_shear_modulus_gpa(material['youngs_modulus_gpa'], material['poisson_reference']):.1f}"
        else:
            g_est = "n/a (aniso)"
        range_text = f"{material['poisson_min']:.2f}-{material['poisson_max']:.2f}"
        print(f"{index:<3} {key:<18} {material['name']:<18} "
              f"{material['youngs_modulus_gpa']:>8.1f} "
              f"{material['poisson_reference']:>7.2f} {range_text:>13} {g_est:>12}")
    print(DASH)
    print("G est = E / (2*(1+nu)) and is shown only for isotropic materials.")
    print("Wood, bamboo and plywood are anisotropic: effective nu is given, E=2G(1+nu) is not applied.")
    print(LINE)


def _ask_float(prompt: str) -> float:
    """Ask for one finite float value."""
    while True:
        try:
            raw = input(prompt)
        except EOFError:
            raise SystemExit("Input ended before all answers were provided.") from None
        try:
            value = float(raw)
        except ValueError:
            print("Please enter a number.")
            continue
        import math

        if not math.isfinite(value):
            print("Please enter a finite number.")
            continue
        return value


def _choose_material() -> str:
    """Ask the user to choose a material by number."""
    keys = list(MATERIAL_DATABASE.keys())
    while True:
        try:
            raw = input(f"Select material 1-{len(keys)}: ")
        except EOFError:
            raise SystemExit("Input ended before a material was selected.") from None
        try:
            choice = int(raw)
        except ValueError:
            print("Please enter an integer.")
            continue
        if 1 <= choice <= len(keys):
            return keys[choice - 1]
        print(f"Please enter a number between 1 and {len(keys)}.")


def _collect_readings(count: int) -> List[Dict[str, float]]:
    """Collect extension/contraction pairs in millimetres."""
    readings: List[Dict[str, float]] = []
    for index in range(1, count + 1):
        print(f"\nReading {index}:")
        while True:
            extension = _ask_float("  axial extension (mm, > 0): ")
            if extension > 0:
                break
            print("  Extension must be positive.")
        contraction = _ask_float("  transverse contraction (mm, positive = thinner): ")
        readings.append({"extension_mm": extension, "contraction_mm": contraction})
    return readings


def show_results(key: str, gauge_length_mm: float, width_mm: float,
                 readings: List[Dict[str, float]], measured_nu: float) -> None:
    """Display a systematic results table."""
    material = MATERIAL_DATABASE[key]
    print("\n" + LINE)
    print("POISSON RATIO RESULTS")
    print(LINE)
    print(f"Material: {material['name']} ({key})")
    print(f"Gauge length: {gauge_length_mm:.3f} mm")
    print(f"Original width: {width_mm:.3f} mm")
    print(f"Reference nu: {material['poisson_reference']:.2f} "
          f"(range {material['poisson_min']:.2f}-{material['poisson_max']:.2f})")
    print(DASH)
    print(f"{'#':<4} {'dL (mm)':>10} {'dw (mm)':>10} {'eps axial':>12} "
          f"{'eps trans':>12} {'nu':>8}")
    print(DASH)
    for index, reading in enumerate(readings, 1):
        extension = reading["extension_mm"]
        contraction = reading["contraction_mm"]
        nu_i = poisson_from_measurement(gauge_length_mm, width_mm, extension, contraction)
        print(f"{index:<4} {extension:>10.4f} {contraction:>10.4f} "
              f"{axial_strain(extension, gauge_length_mm):>12.6f} "
              f"{transverse_strain(contraction, width_mm):>12.6f} {nu_i:>8.3f}")
    print(DASH)
    print(f"Measured nu (ratio of mean strains): {measured_nu:.3f}")
    print(f"Reference nu:                        {material['poisson_reference']:.2f}")
    print(f"Percentage difference:              {percent_difference(measured_nu, material['poisson_reference']):.1f}%")

    verdict = []
    if within_reference_range(measured_nu, key):
        verdict.append("OK: inside the material reference range")
    else:
        verdict.append("CHECK: outside the material reference range")
    if is_stable_poisson(measured_nu):
        verdict.append("OK: inside isotropic stability range (-1 < nu < 0.5)")
    else:
        verdict.append("CHECK: outside isotropic stability range")
    for line in verdict:
        print(line)

    if material["isotropic"] and is_stable_poisson(measured_nu):
        g_est = estimate_shear_modulus_gpa(material["youngs_modulus_gpa"], measured_nu)
        print(f"Estimated shear modulus G from E=2G(1+nu): {g_est:.1f} GPa")
    else:
        print("Shear modulus estimate skipped (anisotropic material or invalid nu).")
    print(LINE)


def _save_results(key: str, gauge_length_mm: float, width_mm: float,
                  readings: List[Dict[str, float]], measured_nu: float) -> None:
    """Save a plain-text report in the current directory."""
    filename = f"poisson_{key}.txt"
    material = MATERIAL_DATABASE[key]
    with open(filename, "w", encoding="utf-8") as handle:
        handle.write(LINE + "\n")
        handle.write("POISSON RATIO RESULTS\n")
        handle.write(LINE + "\n\n")
        handle.write(f"Material: {material['name']} ({key})\n")
        handle.write(f"Gauge length (mm): {gauge_length_mm:.3f}\n")
        handle.write(f"Original width (mm): {width_mm:.3f}\n")
        handle.write(f"Measured nu: {measured_nu:.4f}\n")
        handle.write(f"Reference nu: {material['poisson_reference']:.2f}\n")
        handle.write(f"Difference (%): {percent_difference(measured_nu, material['poisson_reference']):.2f}\n\n")
        for index, reading in enumerate(readings, 1):
            handle.write(
                f"Reading {index}: dL={reading['extension_mm']:.4f} mm, "
                f"dw={reading['contraction_mm']:.4f} mm\n"
            )
    print(f"Saved to {filename}")


def run_interactive() -> None:
    """Run the normal interactive measurement workflow."""
    print(LINE)
    print("POISSON RATIO MEASUREMENT LABORATORY")
    print(LINE)
    print_material_table()

    key = _choose_material()
    gauge_length_mm = _ask_float("Gauge length L0 (mm, > 0): ")
    width_mm = _ask_float("Original transverse width w0 (mm, > 0): ")
    if gauge_length_mm <= 0 or width_mm <= 0:
        raise SystemExit("Gauge length and width must be positive.")

    while True:
        try:
            count = int(input("Number of readings (> 0): "))
        except (ValueError, EOFError):
            print("Please enter a positive integer.")
            continue
        if count > 0:
            break
        print("Please enter a positive integer.")

    readings = _collect_readings(count)
    measured_nu = poisson_from_readings(gauge_length_mm, width_mm, readings)
    show_results(key, gauge_length_mm, width_mm, readings, measured_nu)

    try:
        answer = input("\nSave results to a text file? (yes/no): ").strip().lower()
    except EOFError:
        answer = "no"
    if answer in ("yes", "y"):
        _save_results(key, gauge_length_mm, width_mm, readings, measured_nu)


def run_demo() -> None:
    """Run a fixed, reproducible example without user input."""
    gauge_length_mm = 50.0
    width_mm = 10.0
    readings = [
        {"extension_mm": 0.10, "contraction_mm": 0.006},
        {"extension_mm": 0.20, "contraction_mm": 0.012},
        {"extension_mm": 0.30, "contraction_mm": 0.018},
    ]
    measured_nu = poisson_from_readings(gauge_length_mm, width_mm, readings)
    print("DEMO: mild steel specimen (synthetic data, nu should be about 0.30)")
    show_results("steel", gauge_length_mm, width_mm, readings, measured_nu)


def main(argv: Optional[Sequence[str]] = None) -> None:
    """Program entry point."""
    parser = argparse.ArgumentParser(description="Educational Poisson ratio calculator")
    parser.add_argument("--list", action="store_true", help="print the material reference table")
    parser.add_argument("--demo", action="store_true", help="run a fixed example without input")
    args = parser.parse_args(argv)

    if args.list:
        print_material_table()
    elif args.demo:
        run_demo()
    else:
        run_interactive()
