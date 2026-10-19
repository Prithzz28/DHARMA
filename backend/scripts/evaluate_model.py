#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/evaluate_model.py
==========================
Measure DeFiGuard on *real* labelled transactions, restricted to the 20
features the inference API actually receives. Used to choose the default
``FRAUD_THRESHOLD``.

Two views are reported:
  * receipt — the 20 features as recorded (a confirmed transaction / /predict);
  * pending — receipt-only fields zeroed (cumulative_gas_used, index,
    gas_efficiency), which is what /scan can know before a transaction is mined.

The DeFiTransLyzer CSVs are not part of this repository; point ``--csv-dir``
at the directory containing ``DeFiTransLyzer_fraud.csv`` and
``DeFiTransLyzer_legitimate.csv``.

Usage (from backend/):
    PYTHONPATH=. ../.venv/bin/python scripts/evaluate_model.py --csv-dir /path/to/CSVs --n 2000 \
        --json-out ../frontend/src/data/modelEvaluation.json

``--json-out`` writes the measured figures the dashboard displays (metrics per
threshold, score distribution, outcome counts at the configured threshold,
per-transaction latency and dataset size), so every number on the site traces
back to this script.
"""
from __future__ import annotations

import argparse
import json
import random
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from inference_engine.api.dependencies import AppState
from inference_engine.api.schemas import TransactionFeatures
from inference_engine.core.config import settings
from inference_engine.services.prediction import _sync_predict

API_FEATURES = list(TransactionFeatures.model_fields)
THRESHOLDS = (0.5, 0.65, 0.8, 0.9)


def count_rows(path: Path) -> int:
    """Data rows in a CSV (line count minus the header)."""
    with path.open("rb") as fh:
        return sum(1 for _ in fh) - 1


def sample_rows(path: Path, n: int, seed: int) -> pd.DataFrame:
    """Uniformly sample ~n rows from a large CSV without loading all columns."""
    total = count_rows(path)
    keep = set(random.Random(seed).sample(range(1, total + 1), min(n, total)))
    return pd.read_csv(
        path,
        usecols=lambda c: c in API_FEATURES or c == "flag",
        skiprows=lambda i: i != 0 and i not in keep,
    )


def metrics(y: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    pred = scores > threshold
    tp = int(np.sum(pred & (y == 1)))
    fp = int(np.sum(pred & (y == 0)))
    fn = int(np.sum(~pred & (y == 1)))
    tn = int(np.sum(~pred & (y == 0)))
    return {
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "recall": tp / (tp + fn) if tp + fn else 0.0,
        "fpr": fp / (fp + tn) if fp + tn else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--csv-dir", required=True, type=Path)
    parser.add_argument("--n", type=int, default=2000, help="rows sampled per class")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--json-out", type=Path, help="write dashboard figures to this JSON file")
    args = parser.parse_args()
    warnings.filterwarnings("ignore")

    state = AppState()
    state.load_defiguard(settings.defiguard_model_path)

    fraud_csv = args.csv_dir / "DeFiTransLyzer_fraud.csv"
    legit_csv = args.csv_dir / "DeFiTransLyzer_legitimate.csv"
    df = pd.concat(
        [sample_rows(fraud_csv, args.n, args.seed), sample_rows(legit_csv, args.n, args.seed)],
        ignore_index=True,
    )
    y = df["flag"].to_numpy().astype(int)
    rows = (
        df[[c for c in API_FEATURES if c in df.columns]]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
        .to_dict("records")
    )
    print(f"Sampled {len(rows)} rows ({int(y.sum())} fraud / {int((y == 0).sum())} legitimate)")

    views = {
        "receipt": rows,
        "pending": [dict(r, cumulative_gas_used=0.0, index=0.0, gas_efficiency=0.0) for r in rows],
    }
    model = state.defiguard_model
    threshold = settings.fraud_threshold
    report: dict = {"views": {}}
    print(f"\n{'view':<8} {'thr':>5} {'precision':>9} {'recall':>7} {'FPR':>7}")
    for view, view_rows in views.items():
        scores, latencies = [], []
        for r in view_rows:
            t0 = time.perf_counter()
            scores.append(_sync_predict(model, r, None))
            latencies.append((time.perf_counter() - t0) * 1000)
        scores = np.array(scores)
        rows_out = []
        for thr in sorted(set(THRESHOLDS) | {threshold}):
            m = metrics(y, scores, thr)
            rows_out.append({"threshold": thr, **{k: round(v, 4) for k, v in m.items()}})
            print(f"{view:<8} {thr:>5.2f} {m['precision']:>9.3f} {m['recall']:>7.3f} {m['fpr']:>7.3f}")
        pred = scores > threshold
        bins = np.linspace(0.0, 1.0, 11)
        report["views"][view] = {
            "metrics": rows_out,
            "outcomes": {
                "fraud_caught": int(np.sum(pred & (y == 1))),
                "fraud_missed": int(np.sum(~pred & (y == 1))),
                "legit_flagged": int(np.sum(pred & (y == 0))),
                "legit_passed": int(np.sum(~pred & (y == 0))),
            },
            "accuracy": round(float(np.mean(pred == (y == 1))), 4),
            "score_histogram": [
                {
                    "range": f"{int(lo * 100)}-{int(hi * 100)}%",
                    "fraud": int(np.sum((y == 1) & (scores >= lo) & ((scores < hi) | (hi == 1.0)))),
                    "legit": int(np.sum((y == 0) & (scores >= lo) & ((scores < hi) | (hi == 1.0)))),
                }
                for lo, hi in zip(bins[:-1], bins[1:])
            ],
            "latency_ms": {
                "median": round(float(np.median(latencies)), 2),
                "p95": round(float(np.percentile(latencies, 95)), 2),
            },
        }

    if args.json_out:
        report.update(
            {
                "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
                "generated_by": "backend/scripts/evaluate_model.py",
                "model": "DeFiGuard",
                "threshold": threshold,
                "features": len(API_FEATURES),
                "sample": {"fraud": int(y.sum()), "legit": int((y == 0).sum()), "seed": args.seed},
                # Parsed with pandas: quoted fields can span lines, so a line count would overstate.
                "dataset_rows": {
                    "fraud": len(pd.read_csv(fraud_csv, usecols=["flag"])),
                    "legit": len(pd.read_csv(legit_csv, usecols=["flag"])),
                },
            }
        )
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(report, indent=2) + "\n")
        print(f"\nWrote {args.json_out}")


if __name__ == "__main__":
    main()
