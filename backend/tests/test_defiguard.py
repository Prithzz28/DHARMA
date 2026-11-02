# -*- coding: utf-8 -*-
"""
tests/test_defiguard.py
=======================
DeFiGuard inference must match the training repository's reference predictor
in both modes (pending: pre-execution MLP; receipt: GATv2 receipt graph +
cross-attention), must actually discriminate between inputs, and must never
depend on address identity.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from models.features import NODE_FEATURE_DIM, NODE_TOKEN, TOPIC_TRANSFER, build_ego_graph

GOLDEN = json.loads((Path(__file__).parent / "fixtures" / "defiguard_golden.json").read_text())
REAL = [c for c in GOLDEN["cases"] if c["label"] is not None]


@pytest.mark.parametrize("case", GOLDEN["cases"], ids=lambda c: c["description"])
def test_matches_reference_predictor(dg_model, case):
    assert dg_model.score(case["tx"])["fraud_probability"] == pytest.approx(case["expected_pending"], abs=1e-5)
    if case["receipt"] is not None:
        got = dg_model.score(case["tx"], case["receipt"])["fraud_probability"]
        assert got == pytest.approx(case["expected_receipt"], abs=1e-5)


def test_thresholds_match_reference(dg_model):
    assert dg_model.thresholds == GOLDEN["thresholds"]


@pytest.mark.parametrize("mode", ["pending", "receipt"])
def test_ranks_held_out_fraud_above_legitimate(dg_model, mode):
    """Held-out (never trained on) transactions: better-than-chance ranking."""
    y = [c["label"] for c in REAL]
    scores = [dg_model.score(c["tx"], c["receipt"] if mode == "receipt" else None)["fraud_probability"]
              for c in REAL]
    assert roc_auc_score(y, scores) > 0.6


def test_output_is_not_constant(dg_model):
    scores = [dg_model.score(c["tx"], c["receipt"])["fraud_probability"] for c in REAL]
    assert np.std(scores) > 0.05


def test_networks_rebuilt_strictly_from_state_dicts(dg_model):
    models = dg_model._get_models()
    for mode, stored in (("receipt", dg_model.receipt_state_dict), ("pending", dg_model.pending_state_dict)):
        assert not models[mode].training
        for name, tensor in models[mode].state_dict().items():
            assert np.allclose(tensor.numpy(), stored[name].numpy()), (mode, name)


def test_address_identity_does_not_change_score(dg_model):
    """Rename every address (tx from/to, receipt from/to, log emitters and
    indexed topics) consistently; calldata bytes are left untouched."""
    case = next(c for c in REAL if c["receipt"] and c["receipt"]["logs"])
    addresses = {a.lower() for a in (case["tx"].get("from"), case["tx"].get("to")) if a}
    addresses |= {log["address"].lower() for log in case["receipt"]["logs"]}
    addresses |= {"0x" + t[-40:].lower() for log in case["receipt"]["logs"] for t in log["topics"][1:3]}
    mapping = {a: "0x%040x" % (0xABC000 + i) for i, a in enumerate(sorted(addresses - {"0x" + "0" * 40}))}
    rename = lambda a: mapping.get(a.lower(), a) if isinstance(a, str) else a
    topic = lambda t: "0x" + "0" * 24 + mapping.get("0x" + t[-40:].lower(), "0x" + t[-40:])[2:]
    tx = {**case["tx"], "from": rename(case["tx"]["from"]), "to": rename(case["tx"].get("to"))}
    receipt = {**case["receipt"], "from": rename(case["receipt"].get("from")), "to": rename(case["receipt"].get("to")),
               "logs": [{**log, "address": rename(log["address"]),
                         "topics": log["topics"][:1] + [topic(t) for t in log["topics"][1:3]] + log["topics"][3:]}
                        for log in case["receipt"]["logs"]]}
    assert tx["from"] != case["tx"]["from"]
    for original, renamed in (((case["tx"], None), (tx, None)), ((case["tx"], case["receipt"]), (tx, receipt))):
        a, b = dg_model.score(*original), dg_model.score(*renamed)
        assert a["fraud_probability"] == pytest.approx(b["fraud_probability"], abs=1e-6)


def test_junk_inputs_are_sanitised(dg_model):
    junk = {"value": "inf", "gas": None, "gasPrice": "0xzz", "nonce": [1], "input": 5}
    for receipt in (None, {"gasUsed": "nan", "logs": "x", "status": None}):
        prob = dg_model.score(junk, receipt)["fraud_probability"]
        assert 0.0 <= prob <= 1.0


class TestEgoGraph:
    TX = {"from": "0xAAA", "to": "0xBBB", "value": "0x1"}

    def test_pending_graph_is_sender_tx_recipient(self):
        g = build_ego_graph(self.TX)
        assert list(g["node_types"]) == [0, 1, 1]
        # sender→TX, TX→recipient, each with its reverse edge
        assert g["edge_index"].T.tolist() == [[1, 0], [0, 1], [0, 2], [2, 0]]
        assert g["node_features"].shape == (3, NODE_FEATURE_DIM)

    def test_self_transfer_has_single_account_node(self):
        g = build_ego_graph({"from": "0xAAA", "to": "0xaaa"})
        assert list(g["node_types"]) == [0, 1]

    def test_token_transfer_routes_through_token_node(self):
        a, b, c, d = ("0x" + ch * 40 for ch in "abcd")
        pad = lambda addr: "0x" + "0" * 24 + addr[2:]
        receipt = {"logs": [{"address": c, "topics": [TOPIC_TRANSFER, pad(a), pad(d)], "data": "0x10"}, "junk"]}
        g = build_ego_graph({"from": a, "to": b}, receipt)
        # TX, sender a, recipient b, token c, transfer recipient d
        assert list(g["node_types"]) == [0, 1, 1, NODE_TOKEN, 1]
        # submit, call, emit, a→c and c→d transfer hops — each with a reverse edge
        assert g["edge_index"].shape[1] == 10
        assert [3, 4] in g["edge_index"].T.tolist()
