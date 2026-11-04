# -*- coding: utf-8 -*-
"""Shared fixtures: deterministic settings and isolated global state per test."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from inference_engine.api import dependencies
from inference_engine.core import rate_limit
from inference_engine.core.config import settings
from inference_engine.services import counterparty, explainer, rpc_firewall

BACKEND_DIR = Path(__file__).resolve().parent.parent
DG_MODEL_PATH = BACKEND_DIR / "models" / "defi_guard_model.joblib"


#: Counterparty denylist used by every test (the shipped list is tested separately).
TEST_FLAGGED = ["0x" + "fa" * 20, "0x" + "fb" * 20]


@pytest.fixture(autouse=True)
def _deterministic_settings(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """Pin settings so local .env values never change test outcomes, never call
    the real Groq API (tests opt in with a mock transport), and disable the
    request rate limit unless a test enables it."""
    monkeypatch.setattr(settings, "fraud_threshold", None)  # use the model's per-mode thresholds
    monkeypatch.setattr(settings, "groq_api_key", "")
    monkeypatch.setattr(settings, "groq_model", "openai/gpt-oss-120b")
    monkeypatch.setattr(settings, "llm_max_calls_per_minute", 20)
    monkeypatch.setattr(settings, "rate_limit_per_minute", 0)
    monkeypatch.setattr(explainer, "_groq_transport", None)
    # Wallet RPC firewall: defaults, and never reach a real node (tests inject a mock transport).
    monkeypatch.setattr(settings, "upstream_rpc_url", "https://upstream.test/v3/secret-key")
    monkeypatch.setattr(settings, "rpc_chain_id", 11155111)
    monkeypatch.setattr(settings, "rpc_block_threshold", 1.0)
    flagged_file = tmp_path / "flagged_addresses.json"
    flagged_file.write_text(json.dumps({"addresses": TEST_FLAGGED}))
    monkeypatch.setattr(settings, "flagged_addresses_path", flagged_file)
    monkeypatch.setattr(settings, "rpc_override_window_seconds", 120)
    monkeypatch.setattr(settings, "rpc_rate_limit_per_minute", 0)
    monkeypatch.setattr(rpc_firewall, "_upstream_transport", None)
    monkeypatch.setattr(rpc_firewall, "_upstream_client", None)
    _reset_shared_state()
    yield
    _reset_shared_state()


def _reset_shared_state() -> None:
    explainer._llm_budget.reset()
    rate_limit.request_limiter.reset()
    rpc_firewall.passthrough_limiter.reset()
    rpc_firewall.interception_log.reset()
    rpc_firewall.overrides.reset()
    counterparty.reset_flagged_cache()
    counterparty.reset_code_cache()


@pytest.fixture(autouse=True)
def _isolated_app_state():
    """Snapshot and restore the AppState singleton around every test."""
    state = dependencies.get_app_state()
    saved = copy.copy(state.__dict__)
    yield state
    state.__dict__.clear()
    state.__dict__.update(saved)


@pytest.fixture(scope="session")
def dg_model():
    """The real DeFiGuard model, loaded once per test session."""
    if not DG_MODEL_PATH.exists():
        pytest.skip("DeFiGuard model not present in backend/models.")
    state = dependencies.AppState()
    state.load_defiguard(DG_MODEL_PATH)
    return state.defiguard_model
