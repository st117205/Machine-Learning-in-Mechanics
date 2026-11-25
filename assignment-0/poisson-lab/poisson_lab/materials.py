"""Reference data for the 13 materials used by the educational young-modulus project.

Values are approximate room-temperature data for teaching purposes.
For anisotropic materials (wood, bamboo, plywood) the Poisson ratio is an
effective value along the grain and the isotropic relation E = 2G(1 + nu)
is not applied.
"""

from __future__ import annotations

MATERIAL_DATABASE = {
    "iron": {
        "name": "Iron",
        "youngs_modulus_gpa": 210.0,
        "density_g_cm3": 7.87,
        "poisson_reference": 0.29,
        "poisson_min": 0.27,
        "poisson_max": 0.31,
        "isotropic": True,
    },
    "steel": {
        "name": "Steel (Mild)",
        "youngs_modulus_gpa": 200.0,
        "density_g_cm3": 7.85,
        "poisson_reference": 0.30,
        "poisson_min": 0.28,
        "poisson_max": 0.32,
        "isotropic": True,
    },
    "stainless_steel": {
        "name": "Stainless Steel",
        "youngs_modulus_gpa": 190.0,
        "density_g_cm3": 8.00,
        "poisson_reference": 0.29,
        "poisson_min": 0.27,
        "poisson_max": 0.31,
        "isotropic": True,
    },
    "aluminum": {
        "name": "Aluminum",
        "youngs_modulus_gpa": 69.0,
        "density_g_cm3": 2.70,
        "poisson_reference": 0.33,
        "poisson_min": 0.31,
        "poisson_max": 0.34,
        "isotropic": True,
    },
    "copper": {
        "name": "Copper",
        "youngs_modulus_gpa": 130.0,
        "density_g_cm3": 8.96,
        "poisson_reference": 0.34,
        "poisson_min": 0.32,
        "poisson_max": 0.36,
        "isotropic": True,
    },
    "brass": {
        "name": "Brass",
        "youngs_modulus_gpa": 100.0,
        "density_g_cm3": 8.50,
        "poisson_reference": 0.35,
        "poisson_min": 0.33,
        "poisson_max": 0.37,
        "isotropic": True,
    },
    "oak_wood": {
        "name": "Oak Wood",
        "youngs_modulus_gpa": 11.0,
        "density_g_cm3": 0.75,
        "poisson_reference": 0.40,
        "poisson_min": 0.30,
        "poisson_max": 0.50,
        "isotropic": False,
    },
    "pine_wood": {
        "name": "Pine Wood",
        "youngs_modulus_gpa": 9.0,
        "density_g_cm3": 0.55,
        "poisson_reference": 0.38,
        "poisson_min": 0.30,
        "poisson_max": 0.49,
        "isotropic": False,
    },
    "teak_wood": {
        "name": "Teak Wood",
        "youngs_modulus_gpa": 12.0,
        "density_g_cm3": 0.65,
        "poisson_reference": 0.40,
        "poisson_min": 0.30,
        "poisson_max": 0.50,
        "isotropic": False,
    },
    "bamboo": {
        "name": "Bamboo",
        "youngs_modulus_gpa": 20.0,
        "density_g_cm3": 0.60,
        "poisson_reference": 0.35,
        "poisson_min": 0.25,
        "poisson_max": 0.45,
        "isotropic": False,
    },
    "plywood": {
        "name": "Plywood",
        "youngs_modulus_gpa": 6.0,
        "density_g_cm3": 0.55,
        "poisson_reference": 0.30,
        "poisson_min": 0.20,
        "poisson_max": 0.40,
        "isotropic": False,
    },
    "pvc": {
        "name": "PVC",
        "youngs_modulus_gpa": 3.0,
        "density_g_cm3": 1.40,
        "poisson_reference": 0.40,
        "poisson_min": 0.38,
        "poisson_max": 0.42,
        "isotropic": True,
    },
    "acrylic": {
        "name": "Acrylic",
        "youngs_modulus_gpa": 3.2,
        "density_g_cm3": 1.18,
        "poisson_reference": 0.35,
        "poisson_min": 0.33,
        "poisson_max": 0.38,
        "isotropic": True,
    },
}


def material_keys():
    """Return material keys in a stable order."""
    return list(MATERIAL_DATABASE.keys())
