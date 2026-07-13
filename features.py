"""Feature definitions for Plan B: T2-based 3-class outcome, full macronutrient + spatiotemporal predictors."""
from labels import assign_context_labels, assign_t2_three_class_labels

NUTRIENT_COLS = ["fat_pct", "carbo_pct", "protn_pct", "fat_carbo"]
SPATIOTEMPORAL_COLS = ["Year", "Province"]
FEATURE_COLS = NUTRIENT_COLS + SPATIOTEMPORAL_COLS

FEATURE_NAMES = [
    "fat_energy_ratio",
    "carbo_energy_ratio",
    "protn_energy_ratio",
    "fat_carbo_ratio",
    "Year",
    "Province",
]

NUTRIENT_ALL_COLS = NUTRIENT_COLS
NUTRIENT_ALL_NAMES = ["FatER", "CarbER", "ProtER", "Fat/Carb"]

FEATURE_SETS = {
    "nutrients_only": {
        "cols": NUTRIENT_COLS,
        "names": FEATURE_NAMES[:4],
    },
    "spatiotemporal_only": {
        "cols": SPATIOTEMPORAL_COLS,
        "names": ["Year", "Province"],
    },
    "full": {
        "cols": FEATURE_COLS,
        "names": FEATURE_NAMES,
    },
}

__all__ = [
    "assign_context_labels",
    "assign_t2_three_class_labels",
    "FEATURE_COLS",
    "FEATURE_NAMES",
    "NUTRIENT_COLS",
    "SPATIOTEMPORAL_COLS",
    "NUTRIENT_ALL_COLS",
    "NUTRIENT_ALL_NAMES",
    "FEATURE_SETS",
]
