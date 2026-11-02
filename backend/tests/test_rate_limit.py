# -*- coding: utf-8 -*-
"""Per-client rate limiting on the scoring endpoints."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from inference_engine.api.dependencies import get_app_state
from inference_engine.api.main import app
from inference_engine.core import rate_limit
from inference_engine.core.config import settings
from inference_engine.core.rate_limit import RateLimiter


BODY = {"tx": {"from": "0x" + "1" * 40, "to": "0x" + "2" * 40, "gasPrice": "0x1"}}


def _client(peer: str) -> TestClient:
    return TestClient(app, client=(peer, 50000))


@pytest.fixture(autouse=True)
def _model_and_limit(monkeypatch):
    model = MagicMock()
    model.score.return_value = {"fraud_probability": 0.1, "mode": "pending"}
    model.thresholds = {"pending": 0.5, "receipt": 0.5}
    state = get_app_state()
    state.defiguard_model, state.defiguard_loaded = model, True
    monkeypatch.setattr(settings, "rate_limit_per_minute", 3)


class TestRateLimiter:
    def test_allows_up_to_limit_then_blocks(self):
        limiter = RateLimiter()
        assert [limiter.hit("a", 2)[0] for _ in range(3)] == [True, True, False]

    def test_keys_are_independent(self):
        limiter = RateLimiter()
        limiter.hit("a", 1)
        assert limiter.hit("b", 1)[0] is True

    def test_window_resets(self):
        limiter = RateLimiter(window_seconds=60)
        with patch("inference_engine.core.rate_limit.time.monotonic", return_value=1000.0):
            limiter.hit("a", 1)
            allowed, retry_after = limiter.hit("a", 1)
            assert allowed is False and retry_after == pytest.approx(60)
        with patch("inference_engine.core.rate_limit.time.monotonic", return_value=1060.0):
            assert limiter.hit("a", 1)[0] is True

    def test_zero_limit_allows_nothing(self):
        assert RateLimiter().hit("a", 0)[0] is False


class TestEndpointLimits:
    def test_429_with_retry_after_after_limit(self):
        client = _client("203.0.113.5")
        codes = [client.post("/predict", json=BODY).status_code for _ in range(4)]
        assert codes == [200, 200, 200, 429]
        resp = client.post("/predict", json=BODY)
        assert resp.status_code == 429
        assert 1 <= int(resp.headers["Retry-After"]) <= 60
        assert "Rate limit of 3 requests per minute" in resp.json()["detail"]

    def test_limit_shared_across_scoring_endpoints(self):
        client = _client("203.0.113.6")
        client.post("/predict", json=BODY)
        client.post("/extract", json={"tx_payload": {}})
        client.post("/predict/explain", json={"features": {}})
        assert client.post("/scan", json={"gas_price_gwei": 20}).status_code == 429

    def test_separate_clients_have_separate_budgets(self):
        a, b = _client("203.0.113.7"), _client("203.0.113.8")
        for _ in range(3):
            a.post("/predict", json=BODY)
        assert a.post("/predict", json=BODY).status_code == 429
        assert b.post("/predict", json=BODY).status_code == 200

    def test_health_and_stats_not_limited(self):
        client = _client("203.0.113.9")
        for _ in range(3):
            client.post("/predict", json=BODY)
        assert client.get("/health").status_code == 200
        assert client.get("/stats").status_code == 200

    def test_zero_disables_limiting(self, monkeypatch):
        monkeypatch.setattr(settings, "rate_limit_per_minute", 0)
        client = _client("203.0.113.10")
        assert all(client.post("/predict", json=BODY).status_code == 200 for _ in range(10))


class TestClientIdentification:
    def test_forwarded_for_trusted_from_loopback_proxy(self):
        proxy = _client("127.0.0.1")
        for _ in range(3):
            proxy.post("/predict", json=BODY, headers={"X-Forwarded-For": "198.51.100.1"})
        blocked = proxy.post("/predict", json=BODY, headers={"X-Forwarded-For": "198.51.100.1"})
        other = proxy.post("/predict", json=BODY, headers={"X-Forwarded-For": "198.51.100.2"})
        assert blocked.status_code == 429
        assert other.status_code == 200

    def test_forwarded_for_ignored_from_non_loopback_peer(self):
        client = _client("203.0.113.11")
        for i in range(3):
            client.post("/predict", json=BODY, headers={"X-Forwarded-For": f"198.51.100.{i}"})
        spoofed = client.post("/predict", json=BODY, headers={"X-Forwarded-For": "198.51.100.99"})
        assert spoofed.status_code == 429

    def test_client_key_uses_first_forwarded_hop(self):
        request = MagicMock()
        request.client.host = "::1"
        request.headers = {"x-forwarded-for": "198.51.100.3, 10.0.0.1"}
        assert rate_limit.client_key(request) == "198.51.100.3"
