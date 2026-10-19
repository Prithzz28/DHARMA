#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/build_scanner_presets.py
=================================
Build the dashboard scanner's example transactions from **real labelled
transactions** in the DeFiTransLyzer dataset, so each example's expected
verdict is its ground-truth label rather than an invented scenario.

For each sampled row the scanner form fields are reconstructed (value, gas
price, gas limit, block, log count, sender, tx hash), scored through the real
``POST /scan`` path, and only rows whose verdict matches their label are kept
as presets. The overall agreement rate across all candidates is recorded in
the output, so the selection is transparent.

The dataset has no recipient-address column. A fixed placeholder recipient is
used; it does not affect the score (only its length and whether it equals the
sender are model inputs).

Usage (from backend/):
    PYTHONPATH=. ../.venv/bin/python scripts/build_scanner_presets.py \\
        --csv-dir /path/to/CSVs --out ../frontend/src/data/scannerPresets.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import warnings
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from fastapi.testclient import TestClient

from inference_engine.api.dependencies import get_app_state
from inference_engine.api.main import app
from inference_engine.core.config import settings

COLUMNS = [
    "address", "hash", "flag", "value", "effective_gas_price", "gas_used",
    "block_number", "length_log", "chain_id", "is_same_address", "length_to",
]


def placeholder_recipient(tx_hash: str) -> str:
    """Stand-in recipient derived from the tx hash.

    The dataset has no recipient column; any 42-character address that differs
    from the sender scores identically. A hash-derived address avoids special
    addresses (e.g. 0x…01 is the ecrecover precompile) that an LLM explanation
    could misinterpret.
    """
    return "0x" + hashlib.sha256(tx_hash.encode()).hexdigest()[:40]
#: The scanner estimates gas used as 70% of the gas limit for pending txs.
PENDING_GAS_USED_FRACTION = 0.7


def sample_rows(path: Path, n: int, seed: int) -> pd.DataFrame:
    with path.open("rb") as fh:
        total = sum(1 for _ in fh) - 1
    keep = set(random.Random(seed).sample(range(1, total + 1), min(n, total)))
    return pd.read_csv(path, usecols=COLUMNS, skiprows=lambda i: i != 0 and i not in keep)


def to_form(row: pd.Series) -> dict:
    """Scanner form fields (strings, as typed in the UI) for a dataset row."""
    sender = str(row["address"])
    return {
        "from": sender,
        "to": sender if row["is_same_address"] == 1 else placeholder_recipient(str(row["hash"])),
        "amount": f"{row['value'] / 1e18:.6f}".rstrip("0").rstrip("."),
        "gasPrice": f"{row['effective_gas_price'] / 1e9:.3f}".rstrip("0").rstrip("."),
        "gasLimit": str(math.ceil(row["gas_used"] / PENDING_GAS_USED_FRACTION)),
        "blockNumber": str(int(row["block_number"])),
        "hash": str(row["hash"]),
        "logCount": str(int(row["length_log"])),
    }


def to_scan_request(form: dict) -> dict:
    """Mirror of frontend/src/services/api.js toScanRequest()."""
    return {
        "from": form["from"],
        "to": form["to"],
        "value_eth": float(form["amount"] or 0),
        "gas_price_gwei": float(form["gasPrice"]),
        "gas_limit": int(form["gasLimit"]),
        "block_number": int(form["blockNumber"]),
        "log_count": int(form["logCount"]),
        "tx_hash": form["hash"],
        "explain": False,
    }


def describe(form: dict, fraud: bool) -> str:
    kind = "Fraud" if fraud else "Legit"
    logs = int(form["logCount"])
    action = f"contract call ({logs} logs)" if logs else f"{float(form['amount']):g} ETH transfer"
    if form["from"].lower() == form["to"].lower():
        action = "self-transfer"
    return f"{kind} · {action}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--csv-dir", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--candidates", type=int, default=400, help="rows sampled per class")
    parser.add_argument("--per-class", type=int, default=3)
    parser.add_argument("--seed", type=int, default=29)
    args = parser.parse_args()
    warnings.filterwarnings("ignore")

    settings.rate_limit_per_minute = 0
    get_app_state().load_defiguard(settings.defiguard_model_path)
    client = TestClient(app)

    presets, agreement = [], {}
    for fraud, name in ((True, "DeFiTransLyzer_fraud.csv"), (False, "DeFiTransLyzer_legitimate.csv")):
        df = sample_rows(args.csv_dir / name, args.candidates, args.seed).dropna()
        # Only rows the scanner form can represent faithfully: mainnet, standard
        # recipient address, positive gas figures.
        df = df[(df.chain_id == 1) & (df.length_to == 42) & (df.gas_used > 0) & (df.effective_gas_price > 0)]
        scored = []
        for _, row in df.iterrows():
            form = to_form(row)
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
            by_kind.setdefault(describe(form, fraud).split(" · ")[1].split(" ")[-1], []).append((form, pred))
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
        "threshold": settings.fraud_threshold,
        "source": "Real labelled transactions from the DeFiTransLyzer dataset (sender, hash, value, "
                  "gas, block and log count from the dataset; recipient is a placeholder).",
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
