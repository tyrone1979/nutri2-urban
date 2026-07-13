#!/usr/bin/env python3
"""Re-run entire pipeline for Plan B (T2 3-class outcome, 6-feature predictors)."""
import glob
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(ROOT)


def run_script(script_name: str) -> None:
    path = os.path.join(ROOT, script_name)
    print(f"\n{'=' * 60}\n>>> {script_name}\n{'=' * 60}")
    result = subprocess.run([sys.executable, "-u", path], cwd=ROOT, env={**os.environ, "PYTHONUNBUFFERED": "1"})
    if result.returncode != 0:
        raise SystemExit(f"FAILED: {script_name} (exit {result.returncode})")


def wipe_models() -> None:
    os.makedirs("saved_models", exist_ok=True)
    for pkl in glob.glob("saved_models/*.pkl"):
        os.remove(pkl)
    for pth in glob.glob("saved_models/*.pth"):
        os.remove(pth)
    print("Cleared saved_models/*.pkl and *.pth")


def run_main(force_retrain: bool = True) -> None:
    print(f"\n{'=' * 60}\n>>> main.py (force_retrain={force_retrain})\n{'=' * 60}")
    code = (
        "import multiprocessing\n"
        "try:\n"
        "    multiprocessing.set_start_method('spawn', force=True)\n"
        "except RuntimeError:\n"
        "    pass\n"
        "from main import Trainer\n"
        f"Trainer().run(force_retrain={force_retrain})\n"
    )
    result = subprocess.run([sys.executable, "-c", code], cwd=ROOT)
    if result.returncode != 0:
        raise SystemExit(f"FAILED: main.py (exit {result.returncode})")


STEPS = [
    ("main", None),
    ("leave_one_year_out.py", None),
    ("temporal_validation.py", None),
    ("spatial_validation.py", None),
    ("baseline.py", None),
    ("missing_rate_sensitivity.py", None),
    ("missingness_simulation.py", None),
    ("calibration_analysis.py", None),
    ("bootstrap_analysis.py", None),
    ("downstream_bias_analysis.py", None),
    ("feature_ablation.py", None),
    ("paired_bootstrap_test.py", None),
    ("shap_analysis.py", None),
    ("threshold_sensitivity.py", None),
]


if __name__ == "__main__":
    wipe_models()
    for step, _ in STEPS:
        if step == "main":
            run_main(force_retrain=True)
        else:
            run_script(step)
    print("\nAll experiments complete. Update docx tables from results/*.csv")
