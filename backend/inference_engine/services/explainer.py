# -*- coding: utf-8 -*-
"""
inference_engine/services/explainer.py
========================================
Risk explanation service for the Web3 Transaction Firewall.

Turns the DeFiGuard verdict plus the transaction's feature signals into a
human-readable explanation. The **risk level is always derived from the model
verdict** (``is_fraud`` + ``FRAUD_THRESHOLD``) so the explanation can never
contradict the BLOCKED / SAFE decision; an LLM (Groq, optional) only writes
the reasons and recommendation text. Without a key, when the LLM budget is
exhausted, or when the LLM returns anything malformed, a deterministic
rule-based explainer is used instead.

Response schema
---------------
{
  "risk_level": "low" | "medium" | "high",
  "reasons": ["Reason bullet 1", ...],
  "recommendation": "Actionable recommendation",
  "exec_time_ms": float,
  "is_fallback": bool
}
"""
from __future__ import annotations

import json
import logging
import time
from typing import Any, Dict, List, Optional

import httpx

from inference_engine.core.config import settings
from inference_engine.core.rate_limit import RateLimiter

logger = logging.getLogger(__name__)

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_TIMEOUT_S = 8.0
MAX_REASONS = 6
MAX_TEXT_LEN = 500

#: Probability from which a non-blocked transaction is reported as "medium".
MEDIUM_RISK_FLOOR = 0.5

class LLMUnavailable(Exception):
    """The LLM explanation could not be produced; ``str(exc)`` says why (shown to users)."""


#: Transport override for the Groq HTTP client (tests inject httpx.MockTransport).
_groq_transport: Optional[httpx.AsyncBaseTransport] = None


#: Process-wide LLM call budget (single key).
_llm_budget = RateLimiter()


def mask_address(address: Optional[str]) -> str:
    """Mask an Ethereum address to protect PII (e.g. 0x742d...f44e)."""
    if not address or not isinstance(address, str):
        return "0x..."
    address = address.strip()
    if len(address) <= 10:
        return address
    return f"{address[:6]}...{address[-4:]}"


def risk_level_for(is_fraud: bool, fraud_probability: float) -> str:
    """Risk band consistent with the model verdict."""
    if is_fraud:
        return "high"
    if fraud_probability >= min(MEDIUM_RISK_FLOOR, settings.fraud_threshold):
        return "medium"
    return "low"


def _recommendation_for(risk_level: str) -> str:
    if risk_level == "high":
        return "CRITICAL: Do NOT sign or confirm this transaction. High probability of DeFi exploit or financial loss."
    if risk_level == "medium":
        return "WARNING: Verify contract permissions and target address carefully before proceeding."
    return "Transaction appears safe. Normal baseline parameters detected."


def _build_rule_fallback_explanation(
    features: Dict[str, float],
    is_fraud: bool,
    fraud_probability: float,
    raw_from: Optional[str] = None,
    raw_to: Optional[str] = None,
) -> Dict[str, Any]:
    """Generate structured, deterministic risk explanations based on feature signals."""
    reasons: List[str] = []

    effective_gas_price_gwei = features.get("effective_gas_price", 0.0) / 1e9
    value_eth = features.get("value", 0.0) / 1e18
    is_same_addr = features.get("is_same_address", 0.0) > 0.5
    gas_used = features.get("gas_used", 21000.0)
    log_removed = features.get("log_removed", 0.0) > 0.5

    risk_level = risk_level_for(is_fraud, fraud_probability)

    # Absolute gas price heuristics (no live base fee is available to compare against).
    if effective_gas_price_gwei > 100:
        reasons.append(
            f"Gas price ({effective_gas_price_gwei:.0f} Gwei) is unusually high, a pattern associated with "
            "front-running or urgent drainer transactions."
        )
    elif effective_gas_price_gwei > 50:
        reasons.append(f"Gas price ({effective_gas_price_gwei:.0f} Gwei) is elevated.")

    if value_eth > 10.0:
        reasons.append(f"High ETH transfer volume ({value_eth:.2f} ETH) increases financial risk footprint.")

    if is_same_addr:
        reasons.append(
            f"Self-transfer detected (sender and recipient address match: {mask_address(raw_from)})."
        )

    if log_removed:
        reasons.append("Event logs indicate a blockchain reorganisation or log removal flag.")

    if gas_used > 300_000:
        reasons.append(
            f"High gas consumption ({gas_used:,.0f} units) indicates complex contract execution logic."
        )

    if not reasons and risk_level == "low":
        reasons.append(
            "Transaction parameters (gas price, execution complexity, transfer value) are within normal baseline thresholds."
        )
    elif not reasons:
        reasons.append(
            f"Fraud probability ({fraud_probability * 100:.1f}%) is elevated; no single heuristic "
            "signal explains it — the models flagged the overall feature pattern."
        )

    return {
        "risk_level": risk_level,
        "reasons": reasons,
        "recommendation": _recommendation_for(risk_level),
        "is_fallback": True,
    }


def _validate_llm_payload(parsed: Any) -> Optional[Dict[str, Any]]:
    """Return {reasons, recommendation} if the LLM JSON is well-formed, else None."""
    if not isinstance(parsed, dict):
        return None
    reasons = parsed.get("reasons")
    recommendation = parsed.get("recommendation")
    if not isinstance(reasons, list) or not isinstance(recommendation, str):
        return None
    reasons = [r.strip()[:MAX_TEXT_LEN] for r in reasons if isinstance(r, str) and r.strip()]
    recommendation = recommendation.strip()[:MAX_TEXT_LEN]
    if not reasons or not recommendation:
        return None
    return {"reasons": reasons[:MAX_REASONS], "recommendation": recommendation}


async def _call_groq_explainer(
    features: Dict[str, float],
    is_fraud: bool,
    fraud_probability: float,
    risk_level: str,
    raw_from: Optional[str] = None,
    raw_to: Optional[str] = None,
) -> Dict[str, Any]:
    """Ask Groq for reasons + recommendation.

    Raises ``LLMUnavailable`` with a user-facing reason on any failure.
    """
    api_key = settings.groq_api_key
    if not api_key:
        raise LLMUnavailable("No GROQ_API_KEY configured.")
    allowed, _ = _llm_budget.hit("groq", settings.llm_max_calls_per_minute)
    if not allowed:
        logger.info("LLM call budget exhausted — using rule-based explanation.")
        raise LLMUnavailable("AI explanation limit reached; try again in a minute.")

    value_eth = features.get("value", 0.0) / 1e18
    system_prompt = (
        "You are an expert Web3 DeFi fraud security analyst working for a real-time transaction firewall. "
        "The firewall's DeFiGuard model has ALREADY decided the verdict; explain it, do not contradict it. "
        "Return ONLY a valid JSON object with no Markdown, code fences, or preamble, matching:\n"
        '{"reasons": ["reason 1", "reason 2", "reason 3"], '
        '"recommendation": "clear, actionable recommendation for the wallet owner"}\n\n'
        "Guidelines:\n"
        "- Provide 2-4 specific, technical reasons referencing the actual feature values.\n"
        "- The recommendation must be consistent with the verdict "
        "(BLOCKED: do not sign; SAFE: proceeding is reasonable, with normal caution).\n"
        "- Base every reason ONLY on the values provided. You have no address reputation, "
        "contract source or price data: never claim an address is known, flagged, a burn address "
        "or malicious.\n"
        "- Do NOT include full addresses or other PII."
    )
    user_content = (
        "Ethereum Transaction Risk Analysis:\n"
        f"Verdict: {'BLOCKED' if is_fraud else 'SAFE'} (risk level: {risk_level})\n"
        f"Model Fraud Probability: {fraud_probability * 100:.1f}% "
        f"(block threshold {settings.fraud_threshold * 100:.0f}%)\n"
        f"From Address: {mask_address(raw_from)}\n"
        f"To Address: {mask_address(raw_to)}\n"
        f"Transfer Amount: {value_eth:.6f} ETH\n"
        f"Effective Gas Price: {features.get('effective_gas_price', 0.0) / 1e9:.1f} Gwei\n"
        f"Gas Used: {features.get('gas_used', 0.0):,.0f} units\n"
        f"Self-Transfer (from == to): {features.get('is_same_address', 0.0) > 0.5}\n"
        # The event count is in length_log (log_count is constant 0 in the training data).
        f"Log Events: {features.get('length_log', 0.0):.0f}\n"
    )

    try:
        async with httpx.AsyncClient(timeout=GROQ_TIMEOUT_S, transport=_groq_transport) as client:
            resp = await client.post(
                GROQ_URL,
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json={
                    "model": settings.groq_model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_content},
                    ],
                    "temperature": 0.3,
                    "max_tokens": 500,
                    "response_format": {"type": "json_object"},
                },
            )
        if resp.status_code != 200:
            detail = _groq_error_message(resp)
            if resp.status_code == 404 or (resp.status_code == 400 and "model" in detail.lower()):
                # Typically a retired model: needs a config change, not a retry.
                logger.error("Groq rejected model %r (%d): %s — set GROQ_MODEL to an available model.",
                             settings.groq_model, resp.status_code, detail)
                raise LLMUnavailable(f"Groq model '{settings.groq_model}' is unavailable; set GROQ_MODEL.")
            logger.warning("Groq API returned status %d: %s", resp.status_code, detail)
            raise LLMUnavailable(f"Groq API error (HTTP {resp.status_code}).")
        text = resp.json()["choices"][0]["message"]["content"].strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
        validated = _validate_llm_payload(json.loads(text))
    except httpx.TimeoutException as exc:
        logger.warning("Groq request timed out: %s", exc)
        raise LLMUnavailable("Groq did not respond in time.") from exc
    except httpx.HTTPError as exc:
        logger.warning("Groq request failed (%s: %s)", type(exc).__name__, exc)
        raise LLMUnavailable("Could not reach Groq.") from exc
    except (json.JSONDecodeError, KeyError, IndexError, TypeError, AttributeError) as exc:
        logger.warning("Groq returned an unreadable response (%s: %s)", type(exc).__name__, exc)
        raise LLMUnavailable("Groq returned an unreadable response.") from exc
    if validated is None:
        logger.warning("Groq response did not match the expected schema — using fallback.")
        raise LLMUnavailable("Groq returned an incomplete explanation.")
    return validated


def _groq_error_message(resp: httpx.Response) -> str:
    try:
        return str(resp.json()["error"]["message"])[:300]
    except (ValueError, KeyError, TypeError):
        return resp.text[:300]


async def explain_transaction_risk(
    features: Dict[str, float],
    is_fraud: bool,
    fraud_probability: float,
    raw_from: Optional[str] = None,
    raw_to: Optional[str] = None,
    use_llm: bool = True,
) -> Dict[str, Any]:
    """Generate a structured risk explanation for a scored transaction.

    Tries the Groq LLM first (if enabled and configured), falling back to the
    deterministic rule-based explainer.
    """
    t0 = time.perf_counter()
    risk_level = risk_level_for(is_fraud, fraud_probability)

    fallback_reason: Optional[str] = None
    if use_llm:
        try:
            llm = await _call_groq_explainer(features, is_fraud, fraud_probability, risk_level, raw_from, raw_to)
        except LLMUnavailable as exc:
            fallback_reason = str(exc)
        else:
            return {
                "risk_level": risk_level,
                **llm,
                "is_fallback": False,
                "fallback_reason": None,
                "exec_time_ms": round((time.perf_counter() - t0) * 1000, 2),
            }

    result = _build_rule_fallback_explanation(features, is_fraud, fraud_probability, raw_from, raw_to)
    # Only set when an AI explanation was requested but could not be produced.
    result["fallback_reason"] = fallback_reason
    result["exec_time_ms"] = round((time.perf_counter() - t0) * 1000, 2)
    return result
