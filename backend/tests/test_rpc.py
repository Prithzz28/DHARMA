# -*- coding: utf-8 -*-
"""
tests/test_rpc.py
=================
The wallet RPC firewall (POST /rpc): pass-through, allow / warn / block bands,
the re-send override, fail-closed behaviour, rate limits, batches, and the
info / recent endpoints. A stub model fixes the fraud probability; one test
runs the real DeFiGuard model end to end. The upstream node is a mock.
"""
from __future__ import annotations

import json

import httpx
import pytest
from eth_account import Account
from eth_utils import to_checksum_address
from fastapi.testclient import TestClient

from inference_engine.api.dependencies import get_app_state
from inference_engine.api.main import app
from inference_engine.core.config import settings
from inference_engine.services import rpc_firewall

TO = to_checksum_address("0x8c5d7e12f3a4b5c6d7890abcdef3456789012cd3")
ACCOUNT = Account.from_key("0x" + "22" * 32)
UPSTREAM_HASH = "0x" + "ab" * 32
FLAGGED = "0x" + "fa" * 20            # on the test denylist (conftest.TEST_FLAGGED)
CONTRACT = "0x" + "cc" * 20           # has code on the mock node
WALLET = "0x" + "ee" * 20             # no code on the mock node
MAX_UINT = 2**256 - 1


def approve_data(spender, amount):
    return "0x095ea7b3" + spender[2:].rjust(64, "0") + format(amount, "064x")


def approve_all_data(operator):
    return "0xa22cb465" + operator[2:].rjust(64, "0") + "1".rjust(64, "0")


def signed_tx(value=10**16, data="0x", nonce=0, to=TO):
    signed = ACCOUNT.sign_transaction({
        "to": to_checksum_address(to), "value": value, "gas": 60000, "maxFeePerGas": 20 * 10**9,
        "maxPriorityFeePerGas": 10**9, "nonce": nonce, "chainId": 11155111, "data": data,
    })
    return "0x" + bytes(signed.raw_transaction).hex()


def call(method, params=None, rpc_id=1):
    return {"jsonrpc": "2.0", "id": rpc_id, "method": method, "params": params or []}


def send(raw, rpc_id=1):
    return call("eth_sendRawTransaction", [raw], rpc_id)


class StubModel:
    """Returns a fixed probability; records what it scored."""

    thresholds = {"pending": 0.5, "receipt": 0.5}

    def __init__(self, probability=0.1, error=None):
        self.probability, self.error, self.seen = probability, error, []

    def score(self, tx, receipt=None):
        if self.error:
            raise self.error
        self.seen.append(tx)
        return {"fraud_probability": self.probability, "threshold": 0.5,
                "is_fraud": self.probability >= 0.5, "mode": "pending"}


class Upstream:
    """Mock Ethereum node: answers every call and records what it received."""

    def __init__(self, fail=False, code_lookup_fails=False):
        self.fail, self.code_lookup_fails, self.calls = fail, code_lookup_fails, []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        if self.fail:
            raise httpx.ConnectError("connection refused")
        body = json.loads(request.content)
        self.calls.append(body)
        if body["method"] == "eth_getCode":
            if self.code_lookup_fails:
                return httpx.Response(503, text="unavailable")
            code = "0x6080604052" if body["params"][0].lower() == CONTRACT else "0x"
            return httpx.Response(200, json={"jsonrpc": "2.0", "id": body.get("id"), "result": code})
        result = {"eth_chainId": "0xaa36a7", "eth_blockNumber": "0x10",
                  "eth_sendRawTransaction": UPSTREAM_HASH}.get(body["method"], None)
        return httpx.Response(200, json={"jsonrpc": "2.0", "id": body.get("id"), "result": result})

    @property
    def methods(self):
        return [c["method"] for c in self.calls]


@pytest.fixture
def upstream(monkeypatch):
    node = Upstream()
    monkeypatch.setattr(rpc_firewall, "_upstream_transport", httpx.MockTransport(node))
    return node


@pytest.fixture
def model_can_block(monkeypatch):
    """Let DeFiGuard block on its own (the default, 1.0, only warns)."""
    monkeypatch.setattr(settings, "rpc_block_threshold", 0.7)


def use_model(model):
    state = get_app_state()
    state.defiguard_model, state.defiguard_loaded = model, model is not None
    return model


@pytest.fixture
def client(upstream) -> TestClient:
    return TestClient(app)


def recent(client):
    return client.get("/rpc/recent").json()


class TestPassThrough:
    def test_read_calls_are_forwarded_unscored(self, client, upstream):
        model = use_model(StubModel())
        resp = client.post("/rpc", json=call("eth_chainId", rpc_id=42))
        assert resp.status_code == 200
        assert resp.json() == {"jsonrpc": "2.0", "id": 42, "result": "0xaa36a7"}
        assert upstream.calls == [call("eth_chainId", rpc_id=42)]
        assert model.seen == []

    def test_works_while_the_model_is_not_loaded(self, client, upstream):
        use_model(None)
        assert client.post("/rpc", json=call("eth_blockNumber")).json()["result"] == "0x10"

    def test_batch_skips_notifications(self, client, upstream):
        use_model(StubModel())
        notification = {"jsonrpc": "2.0", "method": "eth_blockNumber", "params": []}
        resp = client.post("/rpc", json=[call("eth_chainId", rpc_id=1), call("eth_blockNumber", rpc_id=2), notification])
        assert sorted(r["id"] for r in resp.json()) == [1, 2]
        assert len(upstream.calls) == 3

    def test_unreachable_upstream_is_a_jsonrpc_error_without_the_url(self, client, monkeypatch):
        monkeypatch.setattr(rpc_firewall, "_upstream_transport", httpx.MockTransport(Upstream(fail=True)))
        use_model(StubModel())
        resp = client.post("/rpc", json=call("eth_chainId"))
        assert resp.status_code == 200
        error = resp.json()["error"]
        assert error["code"] == rpc_firewall.INTERNAL_ERROR
        assert "secret-key" not in resp.text


class TestVerdicts:
    def test_low_risk_transaction_is_forwarded(self, client, upstream):
        model = use_model(StubModel(probability=0.1))
        raw = signed_tx()
        resp = client.post("/rpc", json=send(raw))
        assert resp.json()["result"] == UPSTREAM_HASH
        assert upstream.calls == [send(raw)]  # forwarded byte-for-byte
        assert model.seen[0]["from"] == ACCOUNT.address.lower()
        assert int(model.seen[0]["value"], 16) == 10**16
        item = recent(client)["items"][0]
        assert item["verdict"] == "allowed"
        assert item["tx_hash"] == UPSTREAM_HASH
        assert item["model_signals"] == [] and item["counterparty"] == []

    def test_medium_risk_transaction_is_forwarded_and_flagged(self, client, upstream):
        use_model(StubModel(probability=0.6))
        assert client.post("/rpc", json=send(signed_tx())).json()["result"] == UPSTREAM_HASH
        item = recent(client)["items"][0]
        assert item["verdict"] == "warned"
        assert item["model_signals"]

    def test_high_risk_transaction_is_blocked_and_not_broadcast(self, client, upstream, model_can_block):
        use_model(StubModel(probability=0.9))
        resp = client.post("/rpc", json=send(signed_tx(), rpc_id=7))
        body = resp.json()
        assert resp.status_code == 200 and body["id"] == 7 and "result" not in body
        assert body["error"]["code"] == rpc_firewall.TRANSACTION_REJECTED
        assert "not broadcast" in body["error"]["message"]
        assert body["error"]["data"]["fraud_probability"] == 0.9
        assert body["error"]["data"]["blocked_by"] == "model"
        assert upstream.calls == []
        assert recent(client)["totals"]["blocked"] == 1

    def test_upstream_rejection_is_returned_and_logged(self, client, monkeypatch):
        def node(request):
            body = json.loads(request.content)
            return httpx.Response(200, json={"jsonrpc": "2.0", "id": body["id"],
                                             "error": {"code": -32000, "message": "insufficient funds"}})
        monkeypatch.setattr(rpc_firewall, "_upstream_transport", httpx.MockTransport(node))
        use_model(StubModel(probability=0.1))
        assert client.post("/rpc", json=send(signed_tx())).json()["error"]["message"] == "insufficient funds"
        item = recent(client)["items"][0]
        assert item["upstream_error"] == "insufficient funds" and item["tx_hash"] is None

    def test_eth_send_transaction_is_scored_too(self, client, upstream, model_can_block):
        model = use_model(StubModel(probability=0.9))
        tx = {"from": ACCOUNT.address, "to": TO, "value": "0x1"}
        body = client.post("/rpc", json=call("eth_sendTransaction", [tx])).json()
        assert body["error"]["code"] == rpc_firewall.TRANSACTION_REJECTED
        assert model.seen == [tx]


class TestCounterparty:
    def test_model_alone_only_warns_by_default(self, client, upstream):
        use_model(StubModel(probability=0.99))
        assert client.post("/rpc", json=send(signed_tx())).json()["result"] == UPSTREAM_HASH
        item = recent(client)["items"][0]
        assert item["verdict"] == "warned" and item["blocked_by"] is None

    def test_payment_to_a_flagged_wallet_is_blocked_even_when_the_model_is_calm(self, client, upstream):
        use_model(StubModel(probability=0.01))
        body = client.post("/rpc", json=send(signed_tx(to=FLAGGED))).json()
        assert body["error"]["code"] == rpc_firewall.TRANSACTION_REJECTED
        assert body["error"]["data"]["blocked_by"] == "counterparty"
        assert "labelled fraudulent" in body["error"]["message"]
        assert upstream.calls == []
        item = recent(client)["items"][0]
        assert item["blocked_by"] == "counterparty"
        assert item["counterparty"][0]["code"] == "flagged_counterparty"

    def test_token_transfer_to_a_flagged_wallet_is_blocked(self, client, upstream):
        use_model(StubModel(probability=0.01))
        data = "0xa9059cbb" + FLAGGED[2:].rjust(64, "0") + format(10**18, "064x")
        body = client.post("/rpc", json=send(signed_tx(to=CONTRACT, value=0, data=data))).json()
        assert "token recipient" in body["error"]["message"]

    def test_unlimited_approval_to_a_wallet_is_blocked(self, client, upstream):
        use_model(StubModel(probability=0.01))
        body = client.post("/rpc", json=send(signed_tx(to=CONTRACT, value=0, data=approve_data(WALLET, MAX_UINT)))).json()
        assert body["error"]["data"]["counterparty"][0]["code"] == "approval_to_wallet"
        assert "drainer" in body["error"]["message"]
        assert upstream.methods == ["eth_getCode"]

    def test_approve_all_nfts_to_a_wallet_is_blocked(self, client, upstream):
        use_model(StubModel(probability=0.01))
        body = client.post("/rpc", json=send(signed_tx(to=CONTRACT, value=0, data=approve_all_data(WALLET)))).json()
        assert "every NFT" in body["error"]["message"]

    def test_unlimited_approval_to_a_contract_is_forwarded_with_a_warning(self, client, upstream):
        use_model(StubModel(probability=0.01))
        resp = client.post("/rpc", json=send(signed_tx(to=CONTRACT, value=0, data=approve_data(CONTRACT, MAX_UINT))))
        assert resp.json()["result"] == UPSTREAM_HASH
        item = recent(client)["items"][0]
        assert item["verdict"] == "warned"
        assert item["counterparty"][0]["code"] == "unlimited_approval"
        assert item["model_signals"] == []

    def test_limited_approval_to_a_wallet_warns(self, client, upstream):
        use_model(StubModel(probability=0.01))
        client.post("/rpc", json=send(signed_tx(to=CONTRACT, value=0, data=approve_data(WALLET, 100))))
        assert recent(client)["items"][0]["counterparty"][0]["code"] == "limited_approval_to_wallet"

    def test_revoking_an_approval_is_not_flagged(self, client, upstream):
        use_model(StubModel(probability=0.01))
        client.post("/rpc", json=send(signed_tx(to=CONTRACT, value=0, data=approve_data(WALLET, 0))))
        assert recent(client)["items"][0]["verdict"] == "allowed"

    def test_failed_code_lookup_warns_instead_of_guessing(self, client, monkeypatch):
        node = Upstream(code_lookup_fails=True)
        monkeypatch.setattr(rpc_firewall, "_upstream_transport", httpx.MockTransport(node))
        use_model(StubModel(probability=0.01))
        resp = client.post("/rpc", json=send(signed_tx(to=CONTRACT, value=0, data=approve_data(WALLET, MAX_UINT))))
        assert resp.json()["result"] == UPSTREAM_HASH
        item = recent(client)["items"][0]
        assert item["verdict"] == "warned"
        assert "could not check" in item["counterparty"][0]["message"]

    def test_counterparty_block_can_be_overridden(self, client, upstream):
        use_model(StubModel(probability=0.01))
        raw = signed_tx(to=FLAGGED)
        assert "error" in client.post("/rpc", json=send(raw)).json()
        assert client.post("/rpc", json=send(raw)).json()["result"] == UPSTREAM_HASH
        assert recent(client)["items"][0]["verdict"] == "overridden"


class TestOverride:
    def test_resending_a_blocked_transaction_overrides_once(self, client, upstream, model_can_block):
        use_model(StubModel(probability=0.9))
        raw = signed_tx()
        assert "error" in client.post("/rpc", json=send(raw)).json()
        # The wallet may re-sign with a new nonce / gas — still "the same transaction".
        assert client.post("/rpc", json=send(signed_tx(nonce=1))).json()["result"] == UPSTREAM_HASH
        assert "error" in client.post("/rpc", json=send(raw)).json()  # override consumed
        assert [i["verdict"] for i in recent(client)["items"]] == ["blocked", "overridden", "blocked"]

    def test_a_different_transaction_does_not_override(self, client, upstream, model_can_block):
        use_model(StubModel(probability=0.9))
        client.post("/rpc", json=send(signed_tx(value=1)))
        assert "error" in client.post("/rpc", json=send(signed_tx(value=2))).json()
        assert upstream.calls == []

    def test_override_can_be_disabled(self, client, upstream, monkeypatch, model_can_block):
        monkeypatch.setattr(settings, "rpc_override_window_seconds", 0)
        use_model(StubModel(probability=0.9))
        raw = signed_tx()
        first = client.post("/rpc", json=send(raw)).json()
        assert "override" not in first["error"]["message"]
        assert "error" in client.post("/rpc", json=send(raw)).json()


class TestFailClosed:
    def test_no_model_means_no_broadcast(self, client, upstream):
        use_model(None)
        body = client.post("/rpc", json=send(signed_tx())).json()
        assert body["error"]["code"] == rpc_firewall.RESOURCE_UNAVAILABLE
        assert upstream.calls == []

    def test_scoring_failure_means_no_broadcast(self, client, upstream):
        use_model(StubModel(error=RuntimeError("boom")))
        body = client.post("/rpc", json=send(signed_tx())).json()
        assert body["error"]["code"] == rpc_firewall.INTERNAL_ERROR
        assert upstream.calls == []

    @pytest.mark.parametrize("params", [[], ["0xnothex"], ["0x02c0"], [123]])
    def test_malformed_raw_transaction_is_invalid_params(self, client, upstream, params):
        use_model(StubModel())
        body = client.post("/rpc", json=call("eth_sendRawTransaction", params)).json()
        assert body["error"]["code"] == rpc_firewall.INVALID_PARAMS
        assert upstream.calls == []


class TestProtocol:
    def test_invalid_json(self, client):
        body = client.post("/rpc", content=b"{not json", headers={"Content-Type": "application/json"}).json()
        assert body["error"]["code"] == rpc_firewall.PARSE_ERROR

    @pytest.mark.parametrize("payload", [{"id": 1, "method": "eth_chainId"}, {"jsonrpc": "2.0", "id": 1}, [], [1]])
    def test_invalid_requests(self, client, payload):
        use_model(StubModel())
        body = client.post("/rpc", json=payload).json()
        errors = body if isinstance(body, list) else [body]
        assert all(e["error"]["code"] == rpc_firewall.INVALID_REQUEST for e in errors)


class TestRateLimits:
    def test_transaction_limit_does_not_throttle_wallet_polling(self, client, upstream, monkeypatch):
        monkeypatch.setattr(settings, "rate_limit_per_minute", 1)
        use_model(StubModel(probability=0.1))
        assert "result" in client.post("/rpc", json=send(signed_tx())).json()
        assert client.post("/rpc", json=send(signed_tx(nonce=1))).json()["error"]["code"] == rpc_firewall.LIMIT_EXCEEDED
        for _ in range(5):
            assert "result" in client.post("/rpc", json=call("eth_blockNumber")).json()

    def test_passthrough_limit(self, client, upstream, monkeypatch):
        monkeypatch.setattr(settings, "rpc_rate_limit_per_minute", 2)
        use_model(StubModel())
        codes = [client.post("/rpc", json=call("eth_blockNumber")).json().get("error", {}).get("code")
                 for _ in range(3)]
        assert codes == [None, None, rpc_firewall.LIMIT_EXCEEDED]


class TestInfoAndRecent:
    def test_info_exposes_network_and_policy_but_not_the_upstream_key(self, client):
        use_model(StubModel())
        for path in ("/rpc/info", "/rpc"):
            resp = client.get(path)
            data = resp.json()
            assert data["chain_id"] == 11155111 and data["chain_id_hex"] == "0xaa36a7"
            assert data["upstream_host"] == "upstream.test"
            assert data["warn_threshold"] == 0.5 and data["block_threshold"] == 1.0
            assert data["model_blocks"] is False and data["flagged_addresses"] == 2
            assert "secret-key" not in resp.text

    def test_info_without_model(self, client):
        use_model(None)
        data = client.get("/rpc/info").json()
        assert data["model_loaded"] is False and data["warn_threshold"] is None

    def test_recent_masks_addresses(self, client, upstream):
        use_model(StubModel(probability=0.1))
        client.post("/rpc", json=send(signed_tx()))
        item = recent(client)["items"][0]
        assert item["from_address"] != ACCOUNT.address.lower()
        assert item["from_address"].startswith(ACCOUNT.address.lower()[:6])
        assert "..." in item["to_address"]
        assert item["value_eth"] == pytest.approx(0.01)


def test_real_model_scores_a_signed_wallet_transaction(dg_model, upstream):
    """End to end with DeFiGuard: the decoded raw tx scores like /predict on the same tx."""
    use_model(dg_model)
    client = TestClient(app)
    raw = signed_tx(value=5 * 10**17)
    client.post("/rpc", json=send(raw))
    item = recent(client)["items"][0]
    tx = rpc_firewall.decode_raw_transaction(raw)
    expected = client.post("/predict", json={"tx": tx}).json()["fraud_probability"]
    assert item["fraud_probability"] == pytest.approx(expected)
    assert item["warn_threshold"] == pytest.approx(dg_model.thresholds["pending"])
    assert item["verdict"] in ("allowed", "warned", "blocked")
