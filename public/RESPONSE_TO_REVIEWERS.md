# Response to SiM Major Revision (second round)

## Concern 1 — Binary-task selection bias
- Added explicit discussion of excluding the FatER overlap stratum.
- New sensitivity (Supplementary Table S11): primary exclude Transitional masked acc=0.728;
  collapse-by-T2=0.721; random 50/50=0.667.
- Script: `binary_noise_injection_sensitivity.py`.

## Concern 2 — Three-class presentation
- Abstract now leads with binary 0.716; three-class 0.786 marked descriptive.
- Wording upgraded to **substantially definitional**; Table 3/4 captions footnoted.

## Concern 3 — MAR β₁ choice
- Sensitivity β₁∈{1.0,1.5,2.0,2.5} (Supplementary Table S12): accuracy 0.701–0.741.
- Script: `mar_beta_sensitivity.py`.

## Concern 4 — Adjusted downstream
- Methods now pre-disclose age/sex unavailable; S10 Year+Province adjustment highlighted in Discussion as limited but available adjustment set.
- Unadjusted Table 8 retained as descriptive; adjusted FatER Δcoef≈25%.

## Concern 5 — Hyperparameter sensitivity
- ± neighbourhood grid (Supplementary Table S13): SD=0.001; max |Δ|=0.29 pp.
- Script: `hyperparameter_perturbation.py`.

## Concern 6 — LOYO decline
- Discussion now states decline from 0.795 (1993) to 0.740 (2011) and recommends retraining / time-adaptive weighting (Future Directions).

## Minor
- Spatial masking clarified as observation-level Bernoulli with province-specific rates.
- Table 2 Rural FatER mean≈0.25 explained (admin rural with FatER>30% allowed).
- Retraining moved/expanded under Future Directions.
