#!/usr/bin/env python3
# Run with the DeFiGuard *training* repo venv (it provides the reference predictor):
#   <Dataset>/venv/bin/python -I scripts/generate_defiguard_golden.py <Dataset dir> tests/fixtures/defiguard_golden.json
# Rows whose reference probability is non-finite (float32 overflow in the reference
# implementation) are skipped and counted in the fixture.
"""Generate DeFiGuard parity fixture using the training repo's reference predictor."""
import sys, json, math, warnings
warnings.filterwarnings("ignore")
DATASET = sys.argv[1]; OUT = sys.argv[2]
sys.path.insert(0, DATASET)
import joblib, numpy as np, pandas as pd
p = joblib.load(f"{DATASET}/checkpoints/defi_guard_model.joblib"); p._model = None
feats = list(p.feature_names)

def synth(frm, to, amt, gp, gl, bn, lc, nonce=0):
    gu = gl * 0.7; g = gp * 1e9; v = amt * 1e18
    return {"from": frm, "to": to, "nonce": nonce, "value": v, "gas_used": gu, "effective_gas_price": g,
            "block_number": bn, "chain_id": 1.0, "cumulative_gas_used": 0.0, "length_to": float(len(to)),
            "is_same_address": float(frm.lower() == to.lower()), "gas_efficiency": 0.7, "total_gas_cost": gu * g,
            "gas_price_ratio": 1.0, "log_count": float(lc), "length_log": math.log(lc + 1),
            "event_activity_flag": float(lc > 0), "gas_per_log_event": gu / (lc + 1),
            "normalized_token_transfer": min(v / 1e21, 1.0), "index": 0.0, "log_removed": 0.0}

skipped = []
cases = [
    ("synthetic: self-transfer, high gas", synth("0x742d35Cc6634C0532925a3b844Bc9e7595f44e2a", "0x742d35Cc6634C0532925a3b844Bc9e7595f44e2a", 0, 204, 848000, 1632637, 0), None),
    ("synthetic: plain transfer", synth("0x3b2a1f88e4c5d6a7890bcdef2345678901bcde22", "0x8c5d7e12f3a4b5c6d7890abcdef3456789012cd33", 0.45, 22, 21000, 19000000, 0), None),
    ("synthetic: contract call with logs", synth("0x5555555555555555555555555555555555555555", "0xE592427A0AEce92De3Edee1F18E0157C05861564", 1.5, 28, 145000, 19000015, 4), None),
]
for label, path in ((0, "DeFiTransLyzer_legitimate.csv"), (1, "DeFiTransLyzer_fraud.csv")):
    df = pd.read_csv(f"{DATASET}/CSVs/{path}", nrows=4000).sample(40, random_state=7)
    kept = 0
    for _, row in df.iterrows():
        d = {}
        for f in feats:
            v = row.get(f, 0.0)
            v = float(v) if isinstance(v, (int, float, np.integer, np.floating)) else v
            d[f] = None if (isinstance(v, float) and not math.isfinite(v)) else v
        if kept < 4 and math.isfinite(p.predict(d)["fraud_probability"]):
            cases.append((f"real dataset row (label={label})", d, label)); kept += 1
        elif kept < 4:
            skipped.append(label)

out = []
for desc, tx, label in cases:
    prob = p.predict(tx)["fraud_probability"]
    assert math.isfinite(prob), desc
    out.append({"description": desc, "label": label, "tx": tx, "expected_probability": prob})
json.dump({"source": "DeFiGuard training repo reference predictor (models/predictor.py, PyG 2.8.0.post1)",
           "skipped_nonfinite_reference_rows": len(skipped), "cases": out}, open(OUT, "w"), indent=1, default=float)
for c in out: print(c["description"], c["expected_probability"])
