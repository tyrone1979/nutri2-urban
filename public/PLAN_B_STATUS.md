# Plan B evaluation protocol (non-circular)

## Outcome definition

- **Primary inferential target:** Binary T2 administrative Rural vs Urban (transitional stratum excluded).
- **Three-class labels (descriptive):** T2 Rural/Urban with transitional FatER overlay (23–30%).
- **Predictors:** FatER, CarbER, ProtER, fat/carbo ratio, Year, Province — outcome is not defined solely by predictors.

## Key numbers (30% masked labels)

| Method | Accuracy | Macro-F1 |
|--------|----------|----------|
| Majority | 0.516 | 0.227 |
| MICE | 0.673 | 0.631 |
| KNN | 0.721 | 0.707 |
| LDA | 0.546 | 0.377 |
| RF-Imputer | 0.777 | 0.759 |
| Proposed (BXGB) | 0.786 | 0.770 |

- Holdout three-class accuracy: **0.782**
- Binary (non-transitional) masked accuracy: **0.716**
- Urban class-specific accuracy (masked): **0.421**
- LOYO mean accuracy: **~0.79** (realistic temporal generalisation)
- no-FatER ablation: **0.777** vs full **0.782** (modest drop, not near-perfect either way)

## Regenerate docs

```powershell
py -3 -u run_all_experiments.py
py -3 -u update_reviewer_revisions.py
```
