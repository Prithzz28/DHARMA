# -*- coding: utf-8 -*-
"""
tests/test_defiguard.py
=======================
DeFiGuard inference must match the training repository's reference predictor
(full GATv2 graph encoder + cross-attention fusion), and must actually
discriminate between inputs — guarding against a regression to the old
"tabular-only bypass", which scored every transaction ~0.85.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from models.predictor import build_ego_graph_for_inference

GOLDEN = json.loads((Path(__file__).parent / "fixtures" / "defiguard_golden.json").read_text())


@pytest.mark.parametrize("case", GOLDEN["cases"], ids=lambda c: c["description"])
def test_matches_reference_predictor(dg_model, case):
    assert dg_model.predict_fraud_proba(case["tx"]) == pytest.approx(case["expected_probability"], abs=1e-5)


def test_separates_real_fraud_from_real_legitimate(dg_model):
    real = [c for c in GOLDEN["cases"] if c["label"] is not None]
    fraud = [dg_model.predict_fraud_proba(c["tx"]) for c in real if c["label"] == 1]
    legit = [dg_model.predict_fraud_proba(c["tx"]) for c in real if c["label"] == 0]
    assert fraud and legit
    assert min(fraud) > max(legit)


def test_output_is_not_constant(dg_model):
    scores = [dg_model.predict_fraud_proba(c["tx"]) for c in GOLDEN["cases"]]
    assert np.std(scores) > 0.1


def test_network_rebuilt_strictly_from_state_dict(dg_model):
    model = dg_model.get_model()
    assert not model.training
    for name, tensor in model.state_dict().items():
        assert np.allclose(tensor.numpy(), dg_model.model_state_dict[name].numpy()), name


def test_non_finite_and_overflowing_inputs_are_sanitised(dg_model):
    prob = dg_model.predict_fraud_proba(
        {"value": float("inf"), "gas_used": float("nan"), "erc_20_TokenDivisor": 1e77}
    )
    assert 0.0 <= prob <= 1.0


class TestEgoGraph:
    def test_transfer_graph_structure(self):
        g = build_ego_graph_for_inference({"from": "0xAAA", "to": "0xBBB", "gas_used": 21000})
        assert list(g["node_types"]) == [0, 1, 1]
        # FROM→TX, TX→TO, FROM→TO
        assert g["edge_index"].T.tolist() == [[1, 0], [0, 2], [1, 2]]

    def test_self_transfer_has_single_address_node(self):
        g = build_ego_graph_for_inference({"from": "0xAAA", "to": "0xaaa"})
        assert list(g["node_types"]) == [0, 1]

    def test_feature_only_input_gives_tx_node_only(self):
        g = build_ego_graph_for_inference({"gas_used": 21000})
        assert g["node_features"].shape == (1, 8)
        assert g["edge_index"].shape == (2, 0)

    def test_log_addresses_become_nodes_and_junk_is_ignored(self):
        g = build_ego_graph_for_inference(
            {"from": "0xa", "to": "0xb", "logs": [{"address": "0xc"}, "junk", {"address": "0xc"}]}
        )
        assert len(g["node_types"]) == 4
