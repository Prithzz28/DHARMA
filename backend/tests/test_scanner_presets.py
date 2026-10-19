# -*- coding: utf-8 -*-
"""
The dashboard scanner's example transactions (frontend/src/data/scannerPresets.json)
are real labelled DeFiTransLyzer rows. Each must get the verdict matching its
label through the real /scan path with the real model — if the model, the
feature extraction or FRAUD_THRESHOLD changes, regenerate them with
scripts/build_scanner_presets.py.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from inference_engine.api.dependencies import get_app_state
from inference_engine.api.main import app
from inference_engine.core.config import settings

PRESETS_FILE = Path(__file__).resolve().parents[2] / "frontend" / "src" / "data" / "scannerPresets.json"
PRESETS = json.loads(PRESETS_FILE.read_text())


def to_scan_request(form: dict) -> dict:
    """Mirror of frontend/src/services/api.js toScanRequest() (checked by api.test.js)."""
    return {
        "from": form["from"].strip(),
        "to": form["to"].strip(),
        "value_eth": float(form["amount"] or 0),
        "gas_price_gwei": float(form["gasPrice"]),
        "gas_limit": int(form["gasLimit"]),
        "block_number": int(form["blockNumber"]),
        "log_count": int(form["logCount"]),
        "tx_hash": form["hash"].strip(),
    }


@pytest.fixture
def client(dg_model) -> TestClient:
    state = get_app_state()
    state.defiguard_model, state.defiguard_loaded = dg_model, True
    return TestClient(app)


def test_presets_cover_both_labels():
    labels = {p["expected"] for p in PRESETS["presets"]}
    assert labels == {"BLOCKED", "SAFE"}


def test_presets_generated_for_current_threshold():
    assert PRESETS["threshold"] == settings.fraud_threshold


@pytest.mark.parametrize("preset", PRESETS["presets"], ids=lambda p: p["label"])
def test_preset_verdict_matches_dataset_label(client, preset):
    resp = client.post("/scan", json=to_scan_request(preset["data"]))
    assert resp.status_code == 200, resp.text
    pred = resp.json()["prediction"]
    assert ("BLOCKED" if pred["is_fraud"] else "SAFE") == preset["expected"]
    assert pred["fraud_probability"] == pytest.approx(preset["probability"], abs=1e-5)
