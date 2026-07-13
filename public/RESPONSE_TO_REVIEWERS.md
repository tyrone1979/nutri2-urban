# Response to Reviewer Concerns (SiM revision)

## 1. Transitional circularity (Priority 1)
- **Methods:** Added explicit statement that binary T2 Rural/Urban is the primary inferential target; three-class structure is descriptive only.
- **Abstract/Results:** Transitional accuracy (0.993) reported as exploratory; main evaluation reframed around binary task (accuracy 0.716).
- **Discussion – Limitations:** Circularity listed as first limitation.

## 2. Low Urban accuracy (0.421)
- **Results:** Added paragraph on dietary heterogeneity in urban areas.
- **Discussion:** Reframed framework as screening / probabilistic weighting tool.
- **New analysis:** Urban probability threshold tuning (S8); Urban recall 0.58–0.66 at thresholds 0.40–0.35.

## 3. MICE implementation
- **Methods:** Replaced continuous IterativeImputer with iterative multinomial logistic regression for categorical outcomes.
- **Results:** MICE accuracy updated to 0.673 (from 0.393).
- **Discussion:** Added interpretation regarding non-linear structure.

## 4. Statistical innovation / SiM fit
- **Introduction/Abstract:** Reframed contribution as multi-dimensional **evaluation protocol**, not optimal prediction claim.
- **Title:** Updated to emphasize statistical evaluation protocol.

## 5. Downstream effect size wording
- **Table 8:** Renamed "Bias (%)" to "Relative Change in Cohen's d (%)".
- **Results/Discussion:** "Bias" replaced with "inflation/amplification" language; clarified mechanism via Transitional reassignment.

## 6. New analyses
- **Binary model (S7):** Rural vs. Urban excluding transitional stratum.
- **Threshold tuning (S8):** Application-specific Urban probability cut-offs.

## 7. Cover letter
- Added clarifications on three-category rationale, circularity, Urban accuracy, and SiM scope.

## 8. Second-round revisions
- **Binary metrics:** Added weighted-F1 (0.698) and rural PPV (0.740); reframed as high-specificity rural screening tool.
- **Accuracy comparison:** Explained three-class (0.786) vs binary (0.716) difference.
- **Methods:** Added binary model training paragraph.
- **Discussion:** Consolidated comparison paragraphs; split Strengths/Limitations subsections.
- **Duplicates:** Removed repeated downstream results paragraph.
- **Cover letter:** Binary accuracy stated as primary evaluation metric.
- **Note:** Figure captions (1–5) are present; embedded images should be verified in Word before submission.

## 9. Third-round fine tuning
- **Principal Findings:** Clarified three-class vs binary accuracy wording.
- **Discussion headings:** Standardised `### Strengths` and `### Limitations`; removed blank/orphan paragraphs.
- **Supplementary S7/S8:** Rebuilt as contiguous caption-table blocks.
- **Cover letter:** Removed residual "modest Urban accuracy" phrasing.
