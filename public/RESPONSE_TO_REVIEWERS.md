# Response to SiM Major/Minor Revisions

## Major 1 — Elevating methodological contribution
- Reframed Introduction around **three protocol pillars**: (1) simulated missingness internal validation; (2) separating predictive accuracy from downstream inferential preservation; (3) spatiotemporal generalisability.
- Added a **Protocol application guide** in the Discussion for reuse on other surveys.
- Contribution positioned as a reusable evaluation blueprint, with CHNS as the illustrative testbed (not as an XGBoost methods paper).

## Major 2 — Missingness mechanisms
- New subsection **Missing Data Mechanisms and Simulation**.
- **MCAR**: independent Bernoulli masking with π = 0.30 on the held-out test set.
- **MAR**: logit{P(R=1|FatER)} = β₀ + β₁ z(FatER) with β₁ = 2.0, β₀ = -1.354, realised rate 0.300.
- **Spatial**: observation-level rates 0.50 (Beijing/Shanghai/Chongqing) vs 0.20 otherwise (not province deletion).
- Parameters saved in `results/missingness_mechanism_params.csv`; scripts: `missingness_simulation.py`.

## Major 3 — Metric rationale
- Calibration: framed as enabling **IPW / multiple imputation** reuse of class probabilities, not only fit diagnostics.
- JS divergence: stated as **necessary but not sufficient** (margins ≠ joints); temporal/spatial/downstream checks retained.

## Major 4 — Downstream adjusted analysis
- New analysis: OLS of macronutrients on Urban vs Rural, adjusting for **Year + Province** (age/sex unavailable in diet extract).
- FatER Urban coefficient: true 0.0697 (SE 0.0020) vs imputed 0.0869 (SE 0.0020); relative change 24.6% (Supplementary Table S10).
- Script: `downstream_adjusted_regression.py`.

## Major 5 — Table S9 fragility → primary binary task
- Results now highlight accuracy drop from 0.782 ([0.23,0.30]) to 0.679 ([0.25,0.32]).
- Discussed as evidence that predictor-defined hard thresholds make multi-class accuracy unstable; binary T2 task remains primary (accuracy 0.716).

## Minor
- Fig S1 caption expanded (colour scale; vertical jitter).
- Terminology: label recovery phrased as **imputation/prediction**; “inference” reserved for parameter estimation / protocol pillars where appropriate.
- GitHub (`tyrone1979/nutri2-urban`) includes scripts regenerating S9/S10 and mechanism parameters.
