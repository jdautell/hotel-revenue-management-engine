"""
Corre el pipeline completo: datos -> modelos -> reportes.

    python scripts/run_pipeline.py

Requiere data/hotel_bookings.csv (ver scripts/download_data.py).
"""
import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

for step in ("train_cancellation", "demand", "estimate_elasticity", "insights"):
    print(f"\n{'=' * 70}\n  {step}\n{'=' * 70}")
    runpy.run_path(str(ROOT / "src" / f"{step}.py"), run_name="__main__")
