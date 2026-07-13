#!/usr/bin/env python3
"""Update Supplementary Material.docx for Plan B."""
import shutil
from pathlib import Path

import pandas as pd
from docx import Document

ROOT = Path(__file__).resolve().parent
DOCX = ROOT / "public" / "Supplementary Material.docx"
BACKUP = ROOT / "public" / "Supplementary Material.docx.bak"


def set_cell_text(cell, text: str) -> None:
    if cell.paragraphs:
        cell.paragraphs[0].text = text
        for p in cell.paragraphs[1:]:
            p.text = ""
    else:
        cell.text = text


def replace_para(paragraph, needles, new_text: str) -> bool:
    if any(n in paragraph.text for n in needles):
        paragraph.text = new_text
        return True
    return False


def main():
    shutil.copy2(DOCX, BACKUP)
    doc = Document(str(DOCX))

    spatial = pd.read_csv(ROOT / "results/spatial_validation.csv")
    ablation = pd.read_csv(ROOT / "results/feature_ablation.csv")
    class_acc = pd.read_csv(ROOT / "results/missing_rate_class_acc.csv")
    shap = pd.read_csv(ROOT / "results/shap_main_effects.csv")

    # Table 3 = spatial (index 2 in doc - tables 0-5)
    t3 = doc.tables[2]
    for i, (_, row) in enumerate(spatial.iterrows()):
        r = i + 1
        set_cell_text(t3.rows[r].cells[0], row["Province"])
        set_cell_text(t3.rows[r].cells[1], f"{int(row['n_test']):,}")
        set_cell_text(t3.rows[r].cells[2], f"{row['Accuracy']:.3f}")
        set_cell_text(t3.rows[r].cells[3], f"{row['Macro_F1']:.3f}")
    mean_acc = spatial["Accuracy"].mean()
    sd_acc = spatial["Accuracy"].std()
    mean_f1 = spatial["Macro_F1"].mean()
    sd_f1 = spatial["Macro_F1"].std()
    set_cell_text(t3.rows[13].cells[2], f"{mean_acc:.3f} ± {sd_acc:.3f}")
    set_cell_text(t3.rows[13].cells[3], f"{mean_f1:.3f} ± {sd_f1:.3f}")

    # Table 4 = feature ablation (rewrite structure)
    t4 = doc.tables[3]
    rows = [
        ("Full (6 features)", "full"),
        ("Macronutrients only (4 features)", "nutrients_only"),
        ("Year + Province only (2 features)", "spatiotemporal_only"),
    ]
    for i, (label, key) in enumerate(rows, start=1):
        row = ablation.loc[ablation["Feature_Set"] == key].iloc[0]
        set_cell_text(t4.rows[i].cells[0], label)
        set_cell_text(t4.rows[i].cells[1], f"{row['Accuracy']:.4f}")
        set_cell_text(t4.rows[i].cells[2], f"{row['Macro_F1']:.4f}")
        set_cell_text(t4.rows[i].cells[3], f"{row['Kappa']:.4f}")
        set_cell_text(t4.rows[i].cells[4], "—")

    # Table 5 = missing rate class acc
    t5 = doc.tables[4]
    rates = sorted(class_acc["missing_rate"].unique())
    for i, rate in enumerate(rates, start=1):
        sub = class_acc.loc[class_acc["missing_rate"] == rate]
        rural = sub.loc[sub["class"] == "Rural", "proposed_acc"].iloc[0]
        trans = sub.loc[sub["class"] == "Transitional", "proposed_acc"].iloc[0]
        urban = sub.loc[sub["class"] == "Urban", "proposed_acc"].iloc[0]
        set_cell_text(t5.rows[i].cells[0], f"{int(rate * 100)}%")
        set_cell_text(t5.rows[i].cells[1], f"{rural:.3f}")
        set_cell_text(t5.rows[i].cells[2], f"{trans:.3f}")
        set_cell_text(t5.rows[i].cells[3], f"{urban:.3f}")

    # Table 1 SHAP main effects (global mean |SHAP| only - keep per-class if unavailable)
    t1 = doc.tables[0]
    feat_map = {
        "fat_energy_ratio": "Fat energy ratio",
        "carbo_energy_ratio": "Carbohydrate energy ratio",
        "protn_energy_ratio": "Protein energy ratio",
        "fat_carbo_ratio": "Fat-to-carbohydrate ratio",
        "Year": "Survey year",
        "Province": "Province",
    }
    for i, (_, row) in enumerate(shap.iterrows(), start=1):
        if i >= len(t1.rows):
            break
        name = feat_map.get(row["feature"], row["feature"])
        set_cell_text(t1.rows[i].cells[0], name)
        val = f"{row['main_effect']:.3f}"
        for c in range(1, 5):
            set_cell_text(t1.rows[i].cells[c], val if c == 4 else "—")

    s4_text = (
        "Supplementary Table S4. Feature-set ablation under Plan B (T2 administrative labels with transitional "
        "FatER overlay). Macronutrients and spatiotemporal covariates contributed jointly; year and province alone "
        "performed near chance for three-class inference."
    )
    s5_text = (
        "Supplementary Table S5. Missing rate sensitivity: category-specific accuracy. Transitional accuracy "
        "remained high across missing rates because this stratum is partially defined by FatER thresholds; Urban "
        "accuracy remained near 0.42–0.43, reflecting difficult administrative-urban inference."
    )
    s3_text = (
        "Supplementary Table S3. Leave-one-province-out spatial validation. Mean accuracy "
        f"{mean_acc:.3f} (SD {sd_acc:.3f}). Lower performance in Beijing, Shanghai, and Chongqing reflects "
        "smaller test samples and distinct urban dietary profiles."
    )

    for p in doc.paragraphs:
        replace_para(p, ["Supplementary Table S4."], s4_text)
        replace_para(p, ["Supplementary Table S5."], s5_text)
        replace_para(p, ["Supplementary Table S3."], s3_text)
        replace_para(
            p,
            ["Removing FatER—the feature on which category definitions are based"],
            (
                "Feature-set ablation showed that inference relied on combined macronutrient and spatiotemporal "
                "information rather than any single predictor. Because Transitional labels incorporate FatER "
                "23–30%, high Transitional accuracy is expected and should be interpreted alongside Urban accuracy."
            ),
        )
        replace_para(
            p,
            ["Transitional category accuracy exceeded 0.99 at all missing rates, reflecting the nutritional coherence"],
            (
                "Transitional category accuracy exceeded 0.99 at all missing rates, consistent with partial "
                "definitional overlap between the Transitional stratum and macronutrient predictors."
            ),
        )

    doc.save(str(DOCX))
    print(f"Updated {DOCX}")
    print(f"Backup: {BACKUP}")


if __name__ == "__main__":
    main()
