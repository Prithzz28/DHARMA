#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/build_scanner_presets.py
=================================
Build the dashboard scanner's example transactions from **real labelled,
held-out transactions** of the DeFiGuard research dataset, so each example's
expected verdict is its ground-truth label rather than an invented scenario.

Candidates come from the fold-0 test addresses (never seen in training or
threshold selection). For each, the scanner form fields are taken from the
original transaction (sender, recipient, value, gas price, gas limit, nonce,
calldata), scored through the real ``POST /scan`` path, and only rows whose
verdict matches their label are kept as presets. The agreement rate across
all candidates is recorded in the output, so the selection is transparent.

Usage (from backend/):
    PYTHONPATH=. ../.venv/bin/python scripts/build_scanner_presets.py \\
        --dataset-dir /path/to/Dataset --out ../frontend/src/data/scannerPresets.json
"""
from __future__ import annotations

import argparse
import json
import random
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient

from inference_engine.api.dependencies import get_app_state
from inference_engine.api.main import app
from inference_engine.core.config import settings
from models.features import hex_to_int, split_rpc_result

#: Longest calldata (hex chars) shown in the form — keeps presets readable.
MAX_DATA_CHARS = 2 + 2 * (4 + 32 * 6)


def to_form(tx: dict) -> dict:
    """Scanner form fields (strings, as typed in the UI) for a raw transaction."""
    gas_price = hex_to_int(tx.get("gasPrice")) or hex_to_int(tx.get("maxFeePerGas"))
    data = tx.get("input") or "0x"
    return {
        "from": tx["from"],
        "to": tx["to"],
        "amount": f"{hex_to_int(tx.get('value')) / 1e18:.6f}".rstrip("0").rstrip("."),
        "gasPrice": f"{gas_price / 1e9:.3f}".rstrip("0").rstrip("."),
        "gasLimit": str(hex_to_int(tx.get("gas"))),
        "nonce": str(hex_to_int(tx.get("nonce"))),
        "data": "" if data == "0x" else data,
    }


def to_scan_request(form: dict) -> dict:
    """Mirror of frontend/src/services/api.js toScanRequest()."""
    return {
        "from": form["from"],
        "to": form["to"],
        "value_eth": float(form["amount"] or 0),
        "gas_price_gwei": float(form["gasPrice"]),
        "gas_limit": int(form["gasLimit"]),
        "nonce": int(form["nonce"] or 0),
        "data": form["data"],
        "explain": False,
    }


def describe(form: dict, fraud: bool) -> str:
    kind = "Fraud" if fraud else "Legit"
    if form["from"].lower() == form["to"].lower():
        action = "self-transfer"
    elif form["data"]:
        action = f"contract call ({form['data'][:10]})"
    else:
        action = f"{float(form['amount']):g} ETH transfer"
    return f"{kind} · {action}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset-dir", required=True, type=Path, help="DeFiGuard research repository root")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--candidates", type=int, default=400, help="held-out rows sampled per class")
    parser.add_argument("--per-class", type=int, default=3)
    parser.add_argument("--seed", type=int, default=29)
    args = parser.parse_args()
    warnings.filterwarnings("ignore")

    settings.rate_limit_per_minute = 0
    get_app_state().load_defiguard(settings.defiguard_model_path)
    model = get_app_state().defiguard_model
    client = TestClient(app)

    data = np.load(args.dataset_dir / "data" / "graph_dataset_v2" / "dataset.npz")
    heldout = set(json.loads((args.dataset_dir / "checkpoints" / "heldout_hashes.json").read_text())["hashes"])
    is_heldout = np.isin(data["hashes"], list(heldout))
    rng = np.random.default_rng(args.seed)

    presets, agreement = [], {}
    for fraud in (True, False):
        idx = np.flatnonzero(is_heldout & (data["labels"] == int(fraud)))
        scored = []
        for i in rng.choice(idx, min(args.candidates, len(idx)), replace=False):
            with open(data["paths"][i]) as f:
                tx, _ = split_rpc_result(json.load(f))
            # Only transactions the form can represent: a recipient, a gas price, short calldata.
            if not tx.get("to") or len(tx.get("input") or "0x") > MAX_DATA_CHARS:
                continue
            form = to_form(tx)
            if float(form["gasPrice"] or 0) <= 0:
                continue
            pred = client.post("/scan", json=to_scan_request(form)).json()["prediction"]
            scored.append((form, pred))
        correct = [(f, p) for f, p in scored if p["is_fraud"] == fraud]
        agreement["fraud" if fraud else "legit"] = {
            "candidates": len(scored), "correct": len(correct),
            "rate": round(len(correct) / len(scored), 4) if scored else None,
        }
        # Prefer variety: round-robin over transaction kinds (transfer, contract
        # call, self-transfer), deterministic random order within each kind.
        random.Random(args.seed).shuffle(correct)
        by_kind: dict = {}
        for form, pred in correct:
            by_kind.setdefault(describe(form, fraud).split(" · ")[1].split(" ")[0], []).append((form, pred))
        picked, seen = [], set()
        while len(picked) < args.per_class and any(by_kind.values()):
            for kind in sorted(by_kind):
                while by_kind[kind] and len(picked) < args.per_class:
                    form, pred = by_kind[kind].pop(0)
                    label = describe(form, fraud)
                    if label not in seen:
                        seen.add(label)
                        picked.append((label, form, pred))
                        break
        for label, form, pred in picked:
            presets.append({
                "label": label,
                "expected": "BLOCKED" if fraud else "SAFE",
                "probability": pred["fraud_probability"],
                "data": form,
            })

    out = {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "generated_by": "backend/scripts/build_scanner_presets.py",
        "threshold": model.thresholds["pending"],
        "source": "Real labelled transactions from held-out addresses of the DeFiGuard dataset "
                  "(all form fields taken from the original transaction).",
        "selection": "Rows whose /scan verdict matches their label; agreement rate over all candidates below.",
        "agreement": agreement,
        "presets": presets,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(agreement, indent=2))
    for p in presets:
        print(f"{p['label']:38s} {p['expected']:7s} p={p['probability']:.3f}")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
