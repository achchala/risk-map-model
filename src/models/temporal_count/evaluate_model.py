#!/usr/bin/env python3
"""
Run temporal model training and save metrics for comparison.

Usage:
    python evaluate_model.py

Metrics are saved to outputs/reports/training_metrics.json (append-only).
Lower MAE, RMSE, and Poisson deviance = better.
"""

import json
import logging
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger("evaluate_model")

    # Run training and capture output
    logger.info("Running train_temporal_model.py...")
    result = subprocess.run(
        [sys.executable, "train_temporal_model.py"],
        capture_output=True,
        text=True,
        timeout=1200,  # 20 min (data load + panel + training)
        cwd=Path(__file__).parent,
    )
    stdout = result.stdout + result.stderr
    if result.returncode != 0:
        logger.error("Training failed (exit %d):\n%s", result.returncode, stdout[-3000:])
        return 1

    # Parse metrics from output (look for "Temporal model metrics: MAE=...")
    metrics = {
        "timestamp": datetime.now().isoformat(),
        "mae": None,
        "rmse": None,
        "poisson_deviance": None,
        "with_weather": "Historical weather" in stdout or "Loading historical weather" in stdout,
    }
    for line in stdout.splitlines():
        if "Temporal model metrics:" in line or "Temporal count model evaluation" in line:
            m = re.search(r"MAE=([\d.]+)", line)
            if m: metrics["mae"] = float(m.group(1))
            m = re.search(r"RMSE=([\d.]+)", line)
            if m: metrics["rmse"] = float(m.group(1))
            m = re.search(r"Poisson dev[^=]*=([\d.]+)", line)
            if m: metrics["poisson_deviance"] = float(m.group(1))
            break
    if metrics["mae"] is None:
        logger.warning("Could not parse metrics from output")
    else:
        logger.info("Metrics: MAE=%.4f, RMSE=%.4f, Poisson dev=%.4f",
                    metrics["mae"], metrics["rmse"], metrics["poisson_deviance"])

    # Append to metrics history
    reports_dir = Path("outputs/reports")
    reports_dir.mkdir(parents=True, exist_ok=True)
    metrics_file = reports_dir / "training_metrics.json"
    history = []
    if metrics_file.exists():
        try:
            with open(metrics_file) as f:
                history = json.load(f)
        except json.JSONDecodeError:
            history = []
    if not isinstance(history, list):
        history = [history]
    history.append(metrics)
    with open(metrics_file, "w") as f:
        json.dump(history, f, indent=2)
    logger.info("Saved metrics to %s", metrics_file)

    # Show comparison if we have previous runs
    if len(history) >= 2:
        prev = history[-2]
        curr = history[-1]
        print("\n--- Comparison vs previous run ---")
        print(f"  Previous: with_weather={prev.get('with_weather', 'unknown')}")
        print(f"  Current:  with_weather={curr.get('with_weather', 'unknown')}")
        for k in ("mae", "rmse", "poisson_deviance"):
            if prev.get(k) is not None and curr.get(k) is not None:
                delta = curr[k] - prev[k]
                direction = "better" if delta < 0 else "worse"
                print(f"  {k}: {prev[k]:.4f} → {curr[k]:.4f} ({delta:+.4f}, {direction})")
        print("  (lower = better for all metrics)\n")

    return 0

if __name__ == "__main__":
    sys.exit(main())
