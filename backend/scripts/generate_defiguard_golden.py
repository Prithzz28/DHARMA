#!/usr/bin/env python3
# Run with the DeFiGuard *training* repo venv (it provides the reference predictor):
#   <Dataset>/venv/bin/python -I scripts/generate_defiguard_golden.py <Dataset dir> tests/fixtures/defiguard_golden.json
"""Generate the DeFiGuard parity fixture with the training repo's reference predictor.

Cases: synthetic transactions plus real held-out (fold-0 test address)
transactions, each scored in pending mode (tx only) and receipt mode
(tx + receipt). Receipt fields irrelevant to the model (logsBloom, block
position, signatures) are dropped to keep the fixture small.
"""
import json
import sys
import warnings

warnings.filterwarnings("ignore")
DATASET, OUT = sys.argv[1], sys.argv[2]
sys.path.insert(0, DATASET)
import joblib  # noqa: E402
import numpy as np  # noqa: E402

from models.features import split_rpc_result  # noqa: E402

TX_KEYS = ("from", "to", "value", "gas", "gasPrice", "maxFeePerGas", "nonce", "input")
RECEIPT_KEYS = ("from", "to", "gasUsed", "status", "contractAddress", "logs")
LOG_KEYS = ("address", "topics", "data")

predictor = joblib.load(f"{DATASET}/checkpoints/defi_guard_model.joblib")
data = np.load(f"{DATASET}/data/graph_dataset_v2/dataset.npz")
heldout = set(json.load(open(f"{DATASET}/checkpoints/heldout_hashes.json"))["hashes"])


def slim(tx, receipt):
    tx = {k: tx[k] for k in TX_KEYS if k in tx}
    receipt = {k: receipt[k] for k in RECEIPT_KEYS if k in receipt}
    receipt["logs"] = [{k: log[k] for k in LOG_KEYS if k in log} for log in receipt.get("logs", [])]
    return tx, receipt


def synth(frm, to, eth, gwei, gas, nonce, data="0x"):
    return {"from": frm, "to": to, "value": hex(int(eth * 1e18)), "gas": hex(gas),
            "gasPrice": hex(int(gwei * 1e9)), "nonce": hex(nonce), "input": data}


cases = [
    ("synthetic: plain transfer", synth("0x3b2a1f88e4c5d6a7890bcdef2345678901bcde22",
                                        "0x8c5d7e12f3a4b5c6d7890abcdef3456789012cd3", 0.45, 22, 21000, 57), None, None),
    ("synthetic: fresh-account approval", synth("0x5555555555555555555555555555555555555555",
                                                "0xE592427A0AEce92De3Edee1F18E0157C05861564", 0, 80, 60000, 0,
                                                "0x095ea7b3" + "00" * 64), None, None),
    ("synthetic: self-transfer", synth("0x742d35Cc6634C0532925a3b844Bc9e7595f44e2a",
                                       "0x742d35Cc6634C0532925a3b844Bc9e7595f44e2a", 3, 204, 848000, 0), None, None),
]
rng = np.random.default_rng(7)
for label in (0, 1):
    idx = np.flatnonzero((data["labels"] == label) & np.isin(data["hashes"], list(heldout)))
    for i in rng.choice(idx, 8, replace=False):
        with open(data["paths"][i]) as f:
            tx, receipt = slim(*split_rpc_result(json.load(f)))
        cases.append((f"held-out tx (label={label})", tx, receipt, label))

out = []
for desc, tx, receipt, label in cases:
    entry = {"description": desc, "label": label, "tx": tx, "receipt": receipt,
             "expected_pending": predictor.score(tx)["fraud_probability"]}
    if receipt is not None:
        entry["expected_receipt"] = predictor.score(tx, receipt)["fraud_probability"]
    out.append(entry)

json.dump({"source": "DeFiGuard research repo reference predictor (models/predictor.py)",
           "thresholds": predictor.thresholds, "cases": out}, open(OUT, "w"), indent=1)
for c in out:
    print(f"{c['description']:36s} pending {c['expected_pending']:.4f}  receipt {c.get('expected_receipt', float('nan')):.4f}")
