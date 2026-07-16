#!/usr/bin/env python3
"""Generate Figure 1: SiM statistical evaluation protocol overview (not conference pipeline)."""
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "figures" / "Fig1_framework_overview.png"


def box(ax, x, y, w, h, text, facecolor, fontsize=8.5, weight="normal", edge="#333333"):
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.08",
        linewidth=1.2, edgecolor=edge, facecolor=facecolor,
    )
    ax.add_patch(patch)
    ax.text(
        x + w / 2, y + h / 2, text,
        ha="center", va="center", fontsize=fontsize, fontweight=weight,
        wrap=True, linespacing=1.25,
    )


def arrow(ax, x1, y1, x2, y2):
    ax.add_patch(
        FancyArrowPatch(
            (x1, y1), (x2, y2),
            arrowstyle="-|>", mutation_scale=12,
            linewidth=1.4, color="#444444",
        )
    )


def main():
    fig, ax = plt.subplots(figsize=(11.5, 6.2), dpi=300)
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6.5)
    ax.axis("off")

    # Title band
    ax.text(
        6, 6.25,
        "Statistical Evaluation Protocol for Missing Contextual Label Inference",
        ha="center", va="center", fontsize=11, fontweight="bold", color="#1a1a1a",
    )

    # Row 1: Inputs
    box(ax, 0.3, 4.7, 3.4, 1.15,
        "CHNS cohort (1991–2011)\nn = 101,926\nComplete T2 ground truth",
        "#E8F1F8", fontsize=8)
    box(ax, 4.3, 4.7, 3.4, 1.15,
        "Outcome construction\nPrimary: binary T2 Rural/Urban\nDescriptive: + FatER 23–30%",
        "#F3EDE4", fontsize=8)
    box(ax, 8.3, 4.7, 3.4, 1.15,
        "Predictors (X)\nFatER, CarbER, ProtER, Fat/Carb\nYear, Province",
        "#E9F0E6", fontsize=8)

    arrow(ax, 2.0, 4.7, 2.0, 4.35)
    arrow(ax, 6.0, 4.7, 6.0, 4.35)
    arrow(ax, 10.0, 4.7, 10.0, 4.35)

    # Row 2: Model + masking
    box(ax, 0.8, 3.15, 5.0, 1.05,
        "Class-weighted XGBoost (primary binary + three-class)\nTrain 80% / Test 20%; CV; class weighting",
        "#DCE6F2", fontsize=8.2, weight="bold")
    box(ax, 6.4, 3.15, 5.0, 1.05,
        "Simulated label missingness\nMCAR 10–70%; MAR / spatial scenarios\nMasked labels inferred from X",
        "#F5E6E0", fontsize=8.2, weight="bold")

    arrow(ax, 3.3, 3.15, 3.3, 2.85)
    arrow(ax, 8.9, 3.15, 8.9, 2.85)

    # Row 3: Evaluation suite (SiM differentiator)
    eval_y = 1.35
    eval_h = 1.35
    panels = [
        (0.25, "Missing-rate\nrobustness"),
        (2.55, "Leave-one-\nyear-out"),
        (4.85, "JS divergence\n& calibration"),
        (7.15, "Comparators\n(KNN/MICE/RF)"),
        (9.45, "Downstream\neffect preservation"),
    ]
    for x, label in panels:
        box(ax, x, eval_y, 2.15, eval_h, label, "#FFF8E7", fontsize=7.8)

    # Bracket label
    ax.plot([0.25, 11.6], [1.2, 1.2], color="#8B6914", linewidth=1.5)
    ax.text(
        6, 0.55,
        "Multi-dimensional validation (SiM focus) — primary reporting on binary Rural vs Urban",
        ha="center", va="center", fontsize=8.5, fontstyle="italic", color="#5C4A1F",
    )

    OUT.parent.mkdir(exist_ok=True)
    fig.savefig(OUT, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(OUT.with_suffix(".tiff"), dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
