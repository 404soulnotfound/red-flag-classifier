"""
Quick end-to-end runner:
  1. Generate dataset
  2. Train baseline model
  3. Launch Gradio app

Usage:
    python run.py
"""

import os
import subprocess
import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

BASE = os.path.dirname(__file__)


def run(script: str):
    result = subprocess.run(
        [sys.executable, script],
        cwd=BASE,
        check=True,
    )
    return result


if __name__ == "__main__":
    print("=" * 55)
    print("  🚩 Red Flag Classifier — Full Pipeline Runner")
    print("=" * 55)

    print("\n[1/3] Generating dataset...")
    run(os.path.join(BASE, "data", "generate_dataset.py"))

    print("\n[2/3] Training baseline model...")
    run(os.path.join(BASE, "src", "train_baseline.py"))

    print("\n[3/3] Launching Gradio app → http://localhost:7860")
    run(os.path.join(BASE, "app.py"))
