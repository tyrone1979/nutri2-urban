"""Outcome label definitions — Plan B: T2 administrative + transitional FatER overlay."""
import numpy as np

TRANS_LOW = 0.23
TRANS_HIGH = 0.30


def assign_fater_labels(fat_energy_ratio, trans_low: float = TRANS_LOW, trans_high: float = TRANS_HIGH) -> np.ndarray:
    """Pure FatER thresholds (Plan A reference only — not used as primary outcome)."""
    fat = np.asarray(fat_energy_ratio, dtype=float)
    labels = np.zeros(len(fat), dtype=int)
    labels[(fat >= trans_low) & (fat <= trans_high)] = 1
    labels[fat > trans_high] = 2
    return labels


def assign_t2_three_class_labels(
    t2,
    fat_energy_ratio,
    trans_low: float = TRANS_LOW,
    trans_high: float = TRANS_HIGH,
) -> np.ndarray:
    """
    Plan B outcome: administrative T2 (Urban/Rural) with transitional FatER overlay.

    - Rural (0): T2 = 2 (administrative rural)
    - Urban (2): T2 = 1 (administrative urban)
    - Transitional (1): FatER in [23%, 30%], overriding T2

    Primary inferential target for non-transitional observations is T2-based Rural vs Urban.
    """
    t2 = np.asarray(t2, dtype=int)
    fat = np.asarray(fat_energy_ratio, dtype=float)
    labels = np.zeros(len(t2), dtype=int)
    labels[t2 == 1] = 2  # administrative urban
    labels[(fat >= trans_low) & (fat <= trans_high)] = 1
    return labels


assign_context_labels = assign_t2_three_class_labels
