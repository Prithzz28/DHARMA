# -*- coding: utf-8 -*-
"""
inference_engine/services/explainer.py
========================================
Automated risk explanation service for the Web3 Transaction Firewall.

Gathers structured transaction risk signals, redacts sensitive address data,
and generates human-readable risk explanations via LLM (or rule-based fallback).

LLM Provider: Groq (free tier) using Llama 3 model.
Get your free API key at: https://console.groq.com/keys

Response Schema
---------------
{
  "risk_level": "low" | "medium" | "high",
  "reasons": ["Reason bullet 1", "Reason bullet 2", ...],
  "recommendation": "Actionable security recommendation string",
  "exec_time_ms": float,
  "is_fallback": bool
}
"""
from __future__ import annotations

import json
import logging
import os
import time
from typing import Any, Dict, List, Optional

import httpx

from inference_engine.core.config import settings

logger = logging.getLogger(__name__)


def mask_address(address: Optional[str]) -> str:
    """Mask an Ethereum address to protect PII (e.g. 0x742d...f44e)."""
    if not address or not isinstance(address, str):
        return "0x..."
    address = address.strip()
    if len(address) <= 10:
        return address
    return f"{address[:6]}...{address[-4:]}"


def _build_rule_fallback_explanation(
    features: Dict[str, float],
    is_fraud: bool,
    fraud_probability: float,
    raw_from: Optional[str] = None,
    raw_to: Optional[str] = None,
) -> Dict[str, Any]:
    """Generate structured, deterministic risk explanations based on feature signals."""
    reasons: List[str] = []

    gas_price_ratio = features.get("gas_price_ratio", 1.0)
    effective_gas_price_gwei = features.get("effective_gas_price", 0.0) / 1e9
    value_eth = features.get("value", 0.0) / 1e18
    is_same_addr = features.get("is_same_address", 0.0) > 0.5
    gas_used = features.get("gas_used", 21000.0)
    log_removed = features.get("log_removed", 0.0) > 0.5

    # Determine risk level (aligned with FRAUD_THRESHOLD=0.65)
    if is_fraud or fraud_probability >= 0.65:
        risk_level = "high"
    elif fraud_probability >= 0.50:
        risk_level = "medium"
    else:
        risk_level = "low"

    # Analyze specific risk signals
    if gas_price_ratio > 3.0:
        reasons.append(
            f"Gas price ratio ({gas_price_ratio:.1f}x baseline) is significantly elevated ({effective_gas_price_gwei:.0f} Gwei), "
            "suggesting potential front-running or transaction prioritization manipulation."
        )
    elif gas_price_ratio > 1.8:
        reasons.append(
            f"Gas price ({effective_gas_price_gwei:.0f} Gwei) is higher than network average."
        )

    if value_eth > 10.0:
        reasons.append(
            f"High ETH transfer volume ({value_eth:.2f} ETH) increases financial risk footprint."
        )

    if is_same_addr:
        masked_from = mask_address(raw_from or "0xSame")
        reasons.append(
            f"Self-transfer detected (Sender and Recipient address match: {masked_from})."
        )

    if log_removed:
        reasons.append(
            "Event logs indicate previous blockchain state reorganisation or log removal flag."
        )

    if gas_used > 300_000:
        reasons.append(
            f"High gas consumption ({gas_used:,.0f} units) indicates complex contract execution logic."
        )

    # Low risk default reason
    if not reasons and risk_level == "low":
        reasons.append(
            "Transaction parameters (gas price, execution complexity, transfer value) are within normal baseline thresholds."
        )
    elif not reasons:
        reasons.append(
            f"Random Forest fraud probability score ({fraud_probability * 100:.1f}%) exceeds safety baseline."
        )

    # Recommendation
    if risk_level == "high":
        recommendation = (
            "CRITICAL: Do NOT sign or confirm this transaction. High probability of DeFi exploit or financial loss."
        )
    elif risk_level == "medium":
        recommendation = (
            "WARNING: Verify contract permissions and target address carefully before proceeding."
        )
    else:
        recommendation = (
            "Transaction appears safe. Normal baseline parameters detected."
        )

    return {
        "risk_level": risk_level,
        "reasons": reasons,
        "recommendation": recommendation,
        "is_fallback": True,
    }


async def _call_groq_explainer(
    features: Dict[str, float],
    fraud_probability: float,
    raw_from: Optional[str] = None,
    raw_to: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Call Groq API (Llama 3.3 70B) to generate JSON risk explanation.

    Groq provides free-tier access with very fast inference (~200ms).
    Get a free API key at: https://console.groq.com/keys
    """
    api_key = settings.groq_api_key or os.getenv("GROQ_API_KEY", "")
    if not api_key:
        return None

    masked_from = mask_address(raw_from)
    masked_to = mask_address(raw_to)
    value_eth = features.get("value", 0.0) / 1e18
    gas_price_ratio = features.get("gas_price_ratio", 1.0)
    gas_used = features.get("gas_used", 21000.0)
    gas_efficiency = features.get("gas_efficiency", 0.7)
    is_same_address = features.get("is_same_address", 0.0) > 0.5
    effective_gas_price_gwei = features.get("effective_gas_price", 0.0) / 1e9

    system_prompt = (
        "You are an expert Web3 DeFi fraud security analyst working for a real-time transaction firewall. "
        "Analyze the provided Ethereum transaction feature signals from our Random Forest ML model "
        "and return ONLY a valid JSON object with no Markdown framing, code fences, or preamble.\n\n"
        "The JSON MUST match this exact schema:\n"
        "{\n"
        '  "risk_level": "low" | "medium" | "high",\n'
        '  "reasons": ["detailed plain-language reason 1", "detailed plain-language reason 2", "reason 3"],\n'
        '  "recommendation": "clear, actionable security recommendation for the wallet owner"\n'
        "}\n\n"
        "Guidelines:\n"
        "- Provide 2-4 specific, technical reasons explaining WHY the transaction is risky or safe.\n"
        "- Reference the actual feature values (gas price, self-transfer flag, etc.) in your reasons.\n"
        "- The recommendation must be actionable (e.g., 'Do NOT sign', 'Verify contract source', 'Safe to proceed').\n"
        "- Do NOT include full raw addresses or PII.\n"
        "- Do NOT wrap the JSON in markdown code fences."
    )

    user_content = (
        f"Ethereum Transaction Risk Analysis:\n"
        f"──────────────────────────────────\n"
        f"ML Model Fraud Probability: {fraud_probability * 100:.1f}%\n"
        f"From Address: {masked_from}\n"
        f"To Address: {masked_to}\n"
        f"Transfer Amount: {value_eth:.6f} ETH\n"
        f"Gas Price Ratio (vs base fee): {gas_price_ratio:.2f}x\n"
        f"Effective Gas Price: {effective_gas_price_gwei:.1f} Gwei\n"
        f"Gas Used: {gas_used:,.0f} units\n"
        f"Gas Efficiency (used/limit): {gas_efficiency:.2f}\n"
        f"Self-Transfer (from == to): {is_same_address}\n"
        f"──────────────────────────────────\n"
        f"Provide your expert security analysis as JSON."
    )

    try:
        async with httpx.AsyncClient(timeout=8.0) as client:
            resp = await client.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": "llama-3.3-70b-versatile",
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_content},
                    ],
                    "temperature": 0.3,
                    "max_tokens": 500,
                    "response_format": {"type": "json_object"},
                },
            )
            if resp.status_code == 200:
                data = resp.json()
                text_content = (
                    data.get("choices", [{}])[0]
                    .get("message", {})
                    .get("content", "")
                )
                # Strip potential markdown code fences
                text_content = text_content.strip()
                if text_content.startswith("```"):
                    text_content = text_content.split("\n", 1)[-1]
                if text_content.endswith("```"):
                    text_content = text_content.rsplit("```", 1)[0]
                text_content = text_content.strip()

                parsed = json.loads(text_content)
                # Validate required fields
                if "risk_level" in parsed and "reasons" in parsed and "recommendation" in parsed:
                    parsed["is_fallback"] = False
                    return parsed
                else:
                    logger.warning("Groq response missing required fields: %s", parsed.keys())
            else:
                logger.warning(
                    "Groq API returned status %d: %s",
                    resp.status_code,
                    resp.text[:200],
                )
    except json.JSONDecodeError as exc:
        logger.warning("Failed to parse Groq JSON response: %s", exc)
    except Exception as exc:
        logger.warning("Groq explanation request failed or timed out: %s", exc)

    return None


async def explain_transaction_risk(
    features: Dict[str, float],
    is_fraud: bool,
    fraud_probability: float,
    raw_from: Optional[str] = None,
    raw_to: Optional[str] = None,
) -> Dict[str, Any]:
    """Generate structured risk explanation for a transaction vector.

    Tries Groq LLM first (if GROQ_API_KEY is set), seamlessly falling back
    to the deterministic rule-based explainer.
    """
    t0 = time.perf_counter()

    # 1. Try Groq LLM if configured
    llm_result = await _call_groq_explainer(features, fraud_probability, raw_from, raw_to)
    if llm_result:
        llm_result["exec_time_ms"] = round((time.perf_counter() - t0) * 1000, 2)
        return llm_result

    # 2. Deterministic Rule-Based Fallback
    fallback_result = _build_rule_fallback_explanation(
        features, is_fraud, fraud_probability, raw_from, raw_to
    )
    fallback_result["exec_time_ms"] = round((time.perf_counter() - t0) * 1000, 2)
    return fallback_result
