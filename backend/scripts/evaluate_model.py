#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/evaluate_model.py
==========================
Measure the deployed DeFiGuard model on *held-out* labelled transactions,
through the same code path as the API.

The exported model was trained on fold 0 of the research repository's
address-disjoint protocol; its test addresses were never used for training or
threshold selection. Their transaction hashes ship in
``<dataset>/checkpoints/heldout_hashes.json``; the raw receipts are read from
the research repository's ``data/graph_dataset_v2/dataset.npz`` paths.

Two views are reported:
  * pending — transaction fields only (what /scan knows before signing);
  * receipt — transaction + receipt (full DeFiGuard over the receipt graph).

Usage (from backend/):
    PYTHONPATH=. ../.venv/bin/python scripts/evaluate_model.py --dataset-dir /path/to/Dataset --n 2000 \
        --json-out ../frontend/src/data/modelEvaluation.json

``--json-out`` writes the measured figures the dashboard displays (metrics per
threshold, score distribution, outcome counts at the model's threshold,
per-transaction latency and dataset size), so every number on the site traces
back to this script.
"""
from __future__ import annotations

import argparse
import json
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from inference_engine.api.dependencies import AppState
from inference_engine.core.config import settings
from inference_engine.services.prediction import _sync_predict, threshold_for
from models.features import PRE_EXEC_FEATURES, ALL_FEATURES, split_rpc_result

THRESHOLDS = (0.3, 0.5, 0.7, 0.9)


def load_heldout(dataset_dir: Path, n: int, seed: int):
    """(tx, receipt, label) for up to n held-out transactions per class."""
    data = np.load(dataset_dir / "data" / "graph_dataset_v2" / "dataset.npz")
    heldout = set(json.loads((dataset_dir / "checkpoints" / "heldout_hashes.json").read_text())["hashes"])
    labels, hashes, paths = data["labels"], data["hashes"], data["paths"]
    rng = np.random.default_rng(seed)
    rows = []
    for label in (1, 0):
        idx = np.flatnonzero((labels == label) & np.isin(hashes, list(heldout)))
        for i in rng.choice(idx, min(n, len(idx)), replace=False):
            with open(paths[i]) as f:
                tx, receipt = split_rpc_result(json.load(f))
            rows.append((tx, receipt, int(label)))
    dataset_rows = {"fraud": int(labels.sum()), "legit": int((labels == 0).sum())}
    return rows, dataset_rows


def metrics(y: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    pred = scores >= threshold
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
    parser.add_argument("--dataset-dir", required=True, type=Path, help="DeFiGuard research repository root")
    parser.add_argument("--n", type=int, default=2000, help="held-out rows sampled per class")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--json-out", type=Path, help="write dashboard figures to this JSON file")
    args = parser.parse_args()
    warnings.filterwarnings("ignore")

    state = AppState()
    state.load_defiguard(settings.defiguard_model_path)
    model = state.defiguard_model

    rows, dataset_rows = load_heldout(args.dataset_dir, args.n, args.seed)
    y = np.array([r[2] for r in rows])
    print(f"Held-out sample: {len(rows)} rows ({int(y.sum())} fraud / {int((y == 0).sum())} legitimate)")

    report: dict = {"views": {}}
    print(f"\n{'view':<8} {'thr':>6} {'precision':>9} {'recall':>7} {'FPR':>7}")
    for view in ("pending", "receipt"):
        threshold = threshold_for(model, view)
        scores, latencies = [], []
        for tx, receipt, _ in rows:
            t0 = time.perf_counter()
            scores.append(_sync_predict(model, tx, receipt if view == "receipt" else None)["fraud_probability"])
            latencies.append((time.perf_counter() - t0) * 1000)
        scores = np.array(scores)
        rows_out = []
        for thr in sorted(set(THRESHOLDS) | {threshold}):
            m = metrics(y, scores, thr)
            rows_out.append({"threshold": thr, **{k: round(v, 4) for k, v in m.items()}})
            print(f"{view:<8} {thr:>6.3f} {m['precision']:>9.3f} {m['recall']:>7.3f} {m['fpr']:>7.3f}")
        pred = scores >= threshold
        bins = np.linspace(0.0, 1.0, 11)
        report["views"][view] = {
            "threshold": threshold,
            "features": len(PRE_EXEC_FEATURES if view == "pending" else ALL_FEATURES),
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
                "model": "DeFiGuard v2",
                "evaluation": "held-out addresses (fold 0 of the address-disjoint protocol)",
                # The dashboard shows the pending view (what /scan scores).
                "threshold": report["views"]["pending"]["threshold"],
                "features": report["views"]["pending"]["features"],
                "sample": {"fraud": int(y.sum()), "legit": int((y == 0).sum()), "seed": args.seed},
                "dataset_rows": dataset_rows,
            }
        )
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(report, indent=2) + "\n")
        print(f"\nWrote {args.json_out}")


if __name__ == "__main__":
    main()
