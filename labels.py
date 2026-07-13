import numpy as np
import pandas as pd

RURAL_UPPER = 0.23
TRANS_LOW = 0.23
TRANS_HIGH = 0.30
URBAN_LOWER = 0.30

# CHNS T2: 1 = Urban, 2 = Rural (administrative)


def assign_fater_labels(fat_energy_ratio) -> np.ndarray:
    """Pure FatER thresholds (Plan A — not used for primary outcome)."""
    fat = np.asarray(fat_energy_ratio, dtype=float)
    labels = np.zeros(len(fat), dtype=int)
    labels[(fat >= TRANS_LOW) & (fat <= TRANS_HIGH)] = 1
    labels[fat > URBAN_LOWER] = 2
    return labels


def assign_t2_three_class_labels(
    t2,
    fat_energy_ratio,
    trans_low: float = TRANS_LOW,
    trans_high: float = TRANS_HIGH,
) -> np.ndarray:
    """
    Plan B outcome: administrative T2 (Urban/Rural) with Transitional overlay.

    - Rural (0): T2 = 2 (administrative rural)
    - Urban (2): T2 = 1 (administrative urban)
    - Transitional (1): FatER in [23%, 30%], overriding T2 assignment

    Transitional denotes intermediate dietary composition within either
    administrative stratum, not a separate administrative category.
    """
    t2 = np.asarray(t2, dtype=int)
    fat = np.asarray(fat_energy_ratio, dtype=float)
    labels = np.zeros(len(t2), dtype=int)
    labels[t2 == 1] = 2
    labels[(fat >= trans_low) & (fat <= trans_high)] = 1
    return labels


# Primary outcome for SiM manuscript (Plan B)
assign_context_labels = assign_t2_three_class_labels
