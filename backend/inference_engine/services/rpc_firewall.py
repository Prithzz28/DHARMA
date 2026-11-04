# -*- coding: utf-8 -*-
"""
inference_engine/services/rpc_firewall.py
===========================================
The wallet-facing firewall: a JSON-RPC endpoint users add to their wallet as
a custom RPC URL.

* ``eth_sendRawTransaction`` / ``eth_sendTransaction`` get two independent
  checks: **counterparty checks** (services/counterparty.py — who the
  transaction pays or gives permissions to) and **DeFiGuard** (pending mode —
  no receipt exists before broadcast — scoring how much the transaction
  resembles ones sent by fraud-labelled wallets). The stricter result wins:

  - **allowed**  — no counterparty finding and DeFiGuard below its pending
    threshold: forwarded.
  - **warned**   — a counterparty warning, or DeFiGuard at/above its pending
    threshold: forwarded, and flagged in the interception log.
  - **blocked**  — a counterparty block finding, or DeFiGuard at/above
    ``RPC_BLOCK_THRESHOLD`` (default 1.0: the model alone never blocks):
    refused with a JSON-RPC error (code -32003) the wallet shows to the user.
    Nothing is broadcast.
  - **overridden** — the same blocked transaction (sender, recipient, value,
    calldata) sent again within ``RPC_OVERRIDE_WINDOW_SECONDS``: forwarded.

* Every other method is passed through to ``UPSTREAM_RPC_URL`` unchanged.

The firewall fails closed: if the model is not loaded or scoring fails, the
transaction is refused, never forwarded unscored. Transaction submissions
count against ``RATE_LIMIT_PER_MINUTE`` (like /scan); other calls against the
separate, higher ``RPC_RATE_LIMIT_PER_MINUTE`` so normal wallet polling works.
"""
from __future__ import annotations

import hashlib
import logging
import math
import threading
import time
from collections import deque
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Deque, Dict, List, Literal, Optional
from urllib.parse import urlsplit

import httpx

from feature_extractor.extractor import ExtractionError, extract_features
from inference_engine.api.dependencies import get_app_state
from inference_engine.core.config import settings
from inference_engine.core.rate_limit import RateLimiter, request_limiter
from inference_engine.services.counterparty import check_counterparties, flagged_addresses
from inference_engine.services.explainer import explain_transaction_risk, mask_address
from inference_engine.services.prediction import run_inference
from inference_engine.services.raw_transaction import RawTransactionError, decode_raw_transaction

logger = logging.getLogger(__name__)

SUBMISSION_METHODS = frozenset({"eth_sendRawTransaction", "eth_sendTransaction"})

# JSON-RPC 2.0 / EIP-1474 error codes.
PARSE_ERROR = -32700
INVALID_REQUEST = -32600
INVALID_PARAMS = -32602
INTERNAL_ERROR = -32603
RESOURCE_UNAVAILABLE = -32002
TRANSACTION_REJECTED = -32003
LIMIT_EXCEEDED = -32005

UPSTREAM_TIMEOUT_S = 15.0
LOG_SIZE = 50

Verdict = Literal["allowed", "warned", "blocked", "overridden"]
VERDICTS = ("allowed", "warned", "blocked", "overridden")


def rpc_error(rpc_id: Any, code: int, message: str, data: Any = None) -> Dict[str, Any]:
    """A JSON-RPC 2.0 error response."""
    error: Dict[str, Any] = {"code": code, "message": message}
    if data is not None:
        error["data"] = data
    return {"jsonrpc": "2.0", "id": rpc_id, "error": error}


def upstream_host() -> str:
    """Host name of the upstream node — the URL itself may contain an API key."""
    return urlsplit(settings.upstream_rpc_url).hostname or ""


# --------------------------------------------------------------------------- #
# Interception log (shown on the dashboard's live feed)
# --------------------------------------------------------------------------- #
@dataclass
class Interception:
    id: int
    time: str
    method: str
    verdict: Verdict
    fraud_probability: float
    warn_threshold: float
    block_threshold: float
    from_address: str
    to_address: str
    value_eth: float
    tx_hash: Optional[str] = None
    upstream_error: Optional[str] = None
    #: Counterparty findings: {"severity": "block"|"warn", "code": ..., "message": ...}.
    counterparty: List[Dict[str, str]] = field(default_factory=list)
    #: DeFiGuard's sender-pattern signals (only when it reached its warn threshold).
    model_signals: List[str] = field(default_factory=list)
    #: What caused a block: "counterparty", "model", or None.
    blocked_by: Optional[str] = None


class InterceptionLog:
    """The most recent scored submissions (addresses masked) plus verdict totals."""

    def __init__(self, size: int = LOG_SIZE) -> None:
        self._lock = threading.Lock()
        self._items: Deque[Interception] = deque(maxlen=size)
        self._totals = dict.fromkeys(VERDICTS, 0)
        self._next_id = 1

    def add(self, **fields: Any) -> Interception:
        with self._lock:
            item = Interception(
                id=self._next_id, time=datetime.now(timezone.utc).isoformat(timespec="seconds"), **fields
            )
            self._next_id += 1
            self._items.appendleft(item)
            self._totals[item.verdict] += 1
            return item

    def snapshot(self, limit: int) -> Dict[str, Any]:
        with self._lock:
            items = [asdict(item) for item in list(self._items)[:limit]]
            return {"items": items, "totals": dict(self._totals)}

    def reset(self) -> None:
        with self._lock:
            self._items.clear()
            self._totals = dict.fromkeys(VERDICTS, 0)
            self._next_id = 1


interception_log = InterceptionLog()


# --------------------------------------------------------------------------- #
# Override: a blocked transaction sent again within the window is forwarded
# --------------------------------------------------------------------------- #
def override_key(tx: Dict[str, Any]) -> str:
    """Identify "the same transaction" independent of nonce and gas, which a
    wallet may re-estimate when the user retries."""
    parts = [str(tx.get(name) or "").lower() for name in ("from", "to", "value")]
    parts.append(str(tx.get("input") or tx.get("data") or "0x").lower())
    return hashlib.sha256("|".join(parts).encode()).hexdigest()


class OverrideRegistry:
    """Blocked-transaction keys with the time they were blocked."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._blocked: Dict[str, float] = {}

    def consume(self, key: str, window_s: float) -> bool:
        """True (and forget the key) if *key* was blocked within *window_s*."""
        now = time.monotonic()
        with self._lock:
            blocked_at = self._blocked.pop(key, None)
            return blocked_at is not None and now - blocked_at <= window_s

    def arm(self, key: str, window_s: float) -> None:
        now = time.monotonic()
        with self._lock:
            if len(self._blocked) > 10_000:
                self._blocked = {k: t for k, t in self._blocked.items() if now - t <= window_s}
            self._blocked[key] = now

    def reset(self) -> None:
        with self._lock:
            self._blocked.clear()


overrides = OverrideRegistry()

#: Per-client limiter for non-transaction RPC calls.
passthrough_limiter = RateLimiter()


# --------------------------------------------------------------------------- #
# Upstream node
# --------------------------------------------------------------------------- #
#: Transport override for the upstream client (tests inject httpx.MockTransport).
_upstream_transport: Optional[httpx.AsyncBaseTransport] = None
_upstream_client: Optional[httpx.AsyncClient] = None


def _client() -> httpx.AsyncClient:
    global _upstream_client
    if _upstream_client is None or _upstream_client.is_closed:
        _upstream_client = httpx.AsyncClient(timeout=UPSTREAM_TIMEOUT_S, transport=_upstream_transport)
    return _upstream_client


async def close_upstream_client() -> None:
    global _upstream_client
    if _upstream_client is not None and not _upstream_client.is_closed:
        await _upstream_client.aclose()
    _upstream_client = None


async def forward(call: Dict[str, Any]) -> Dict[str, Any]:
    """Send one JSON-RPC call to the upstream node and return its response."""
    rpc_id = call.get("id")
    if not settings.upstream_rpc_url:
        return rpc_error(rpc_id, RESOURCE_UNAVAILABLE, "No upstream RPC node is configured (UPSTREAM_RPC_URL).")
    try:
        resp = await _client().post(settings.upstream_rpc_url, json=call)
    except httpx.TimeoutException:
        logger.warning("Upstream RPC %s timed out (%s).", upstream_host(), call.get("method"))
        return rpc_error(rpc_id, INTERNAL_ERROR, "Upstream RPC node timed out.")
    except httpx.HTTPError as exc:
        logger.warning("Upstream RPC %s unreachable: %s", upstream_host(), type(exc).__name__)
        return rpc_error(rpc_id, INTERNAL_ERROR, "Upstream RPC node is unreachable.")
    try:
        body = resp.json()
    except ValueError:
        body = None
    if isinstance(body, dict) and ("result" in body or "error" in body):
        body["jsonrpc"], body["id"] = "2.0", rpc_id
        return body
    logger.warning("Upstream RPC %s answered HTTP %d without a JSON-RPC body.", upstream_host(), resp.status_code)
    return rpc_error(rpc_id, INTERNAL_ERROR, f"Upstream RPC node returned HTTP {resp.status_code}.")


async def has_code(address: str) -> Optional[bool]:
    """Whether *address* holds contract code on the upstream chain (None if the lookup failed)."""
    response = await forward({"jsonrpc": "2.0", "id": "defiguard-getcode", "method": "eth_getCode",
                              "params": [address, "latest"]})
    code = response.get("result")
    if not isinstance(code, str):
        return None
    return code.lower() not in ("0x", "0x0", "")


# --------------------------------------------------------------------------- #
# Request handling
# --------------------------------------------------------------------------- #
def _tx_from_params(method: str, params: Any) -> Dict[str, Any]:
    if not isinstance(params, list) or not params:
        raise RawTransactionError(f"{method} expects a non-empty params array.")
    if method == "eth_sendRawTransaction":
        return decode_raw_transaction(params[0])
    if not isinstance(params[0], dict):
        raise RawTransactionError("eth_sendTransaction expects a transaction object.")
    return params[0]


def _value_eth(tx: Dict[str, Any]) -> float:
    try:
        return int(str(tx.get("value") or "0x0"), 16) / 1e18
    except ValueError:
        return 0.0


def _block_message(blocked_by: str, probability: float, block_threshold: float, reason: Optional[str]) -> str:
    if blocked_by == "counterparty":
        message = f"DeFiGuard firewall blocked this transaction: {reason} It was not broadcast."
    else:
        message = (
            f"DeFiGuard firewall blocked this transaction: sender-pattern risk {probability:.0%} "
            f"(block threshold {block_threshold:.0%}). It was not broadcast."
        )
        if reason:
            message += f" Main signal: {reason}"
    window = settings.rpc_override_window_seconds
    if window:
        message += f" If you trust it, send the same transaction again within {window}s to override."
    return message


async def _handle_submission(call: Dict[str, Any], client_id: str, model: Optional[Any]) -> Dict[str, Any]:
    rpc_id, method = call.get("id"), call["method"]

    limit = settings.rate_limit_per_minute
    if limit > 0:
        allowed, retry_after = request_limiter.hit(client_id, limit)
        if not allowed:
            seconds = max(1, math.ceil(retry_after))
            return rpc_error(rpc_id, LIMIT_EXCEEDED,
                             f"Rate limit of {limit} transactions per minute exceeded. Retry in {seconds}s.")

    if model is None:
        return rpc_error(rpc_id, RESOURCE_UNAVAILABLE,
                         "DeFiGuard firewall is unavailable, so the transaction was not sent. Try again later.")

    try:
        tx = _tx_from_params(method, call.get("params"))
        prediction = await run_inference(model, tx)
        features = extract_features(tx)
        report = await check_counterparties(tx, has_code, flagged_addresses(settings.flagged_addresses_path))
    except (RawTransactionError, ExtractionError) as exc:
        return rpc_error(rpc_id, INVALID_PARAMS, str(exc))
    except Exception:  # noqa: BLE001 — fail closed: never forward an unscored transaction
        logger.exception("Scoring a %s failed; refusing the transaction.", method)
        return rpc_error(rpc_id, INTERNAL_ERROR,
                         "DeFiGuard could not score this transaction, so it was not sent.")
    get_app_state().record_prediction(prediction.is_fraud, prediction.exec_time_ms)

    probability, warn_threshold = prediction.fraud_probability, prediction.threshold
    block_threshold = settings.rpc_block_threshold
    model_blocks = block_threshold < 1.0 and probability >= block_threshold
    model_warns = probability >= warn_threshold
    blocked_by = "counterparty" if report.blocks else "model" if model_blocks else None
    verdict: Verdict = (
        "blocked" if blocked_by
        else "warned" if report.warns or model_warns
        else "allowed"
    )

    model_signals: List[str] = []
    if model_warns or model_blocks:
        explanation = await explain_transaction_risk(
            features=features, is_fraud=prediction.is_fraud, fraud_probability=probability,
            threshold=warn_threshold, raw_from=tx.get("from"), raw_to=tx.get("to"), use_llm=False,
        )
        model_signals = explanation["reasons"]
    findings = [{"severity": f.severity, "code": f.code, "message": f.message} for f in report.findings]

    record = dict(
        method=method, fraud_probability=probability, warn_threshold=warn_threshold,
        block_threshold=block_threshold, from_address=mask_address(tx.get("from")),
        to_address=mask_address(tx.get("to")) if tx.get("to") else "contract creation",
        value_eth=_value_eth(tx), counterparty=findings, model_signals=model_signals,
        blocked_by=blocked_by,
    )

    window = settings.rpc_override_window_seconds
    if blocked_by:
        key = override_key(tx)
        if window and overrides.consume(key, window):
            verdict = "overridden"
        else:
            if window:
                overrides.arm(key, window)
            interception_log.add(verdict="blocked", **record)
            logger.info("RPC firewall BLOCKED %s (%s) p=%.3f", method, blocked_by, probability)
            reason = (
                next(f.message for f in report.findings if f.severity == "block") if blocked_by == "counterparty"
                else model_signals[0] if model_signals else None
            )
            return rpc_error(
                rpc_id, TRANSACTION_REJECTED,
                _block_message(blocked_by, probability, block_threshold, reason),
                data={"blocked_by": blocked_by, "counterparty": findings,
                      "fraud_probability": probability, "block_threshold": block_threshold,
                      "model_signals": model_signals, "override_window_seconds": window},
            )

    response = await forward(call)
    result, error = response.get("result"), response.get("error")
    interception_log.add(
        verdict=verdict,
        tx_hash=result if isinstance(result, str) else None,
        upstream_error=str(error.get("message", ""))[:200] if isinstance(error, dict) else None,
        **record,
    )
    logger.info("RPC firewall %s %s p=%.3f", verdict.upper(), method, probability)
    return response


async def handle_call(call: Any, client_id: str, model: Optional[Any]) -> Optional[Dict[str, Any]]:
    """Handle one JSON-RPC call. Returns None for notifications (no ``id``)."""
    if not isinstance(call, dict) or call.get("jsonrpc") != "2.0" or not isinstance(call.get("method"), str):
        rpc_id = call.get("id") if isinstance(call, dict) else None
        return rpc_error(rpc_id, INVALID_REQUEST, "Invalid JSON-RPC 2.0 request.")

    if call["method"] in SUBMISSION_METHODS:
        response = await _handle_submission(call, client_id, model)
    else:
        limit = settings.rpc_rate_limit_per_minute
        allowed = limit <= 0 or passthrough_limiter.hit(client_id, limit)[0]
        response = (
            await forward(call) if allowed
            else rpc_error(call.get("id"), LIMIT_EXCEEDED, f"Rate limit of {limit} RPC calls per minute exceeded.")
        )
    return response if "id" in call else None
