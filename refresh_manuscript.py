#!/usr/bin/env python3
"""Refresh SiM manuscript: tables, text, figures, supplementary (Plan B)."""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def run(script: str) -> None:
    print(f"\n>>> {script}", flush=True)
    r = subprocess.run([sys.executable, "-u", str(ROOT / script)], cwd=ROOT)
    if r.returncode != 0:
        raise SystemExit(f"Failed: {script}")


if __name__ == "__main__":
    run("update_main_docx.py")
    run("update_reviewer_revisions.py")
    run("update_supplementary_docx.py")
    run("generate_fig1.py")
    run("embed_manuscript_figures.py")  # must be last: earlier steps must not strip image paragraphs
    print("\nManuscript refresh complete.", flush=True)
