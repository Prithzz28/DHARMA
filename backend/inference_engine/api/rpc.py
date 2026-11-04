# -*- coding: utf-8 -*-
"""
inference_engine/api/rpc.py
=============================
Wallet RPC firewall routes.

POST /rpc          JSON-RPC 2.0 endpoint (single call or batch). Add its URL
                   to a wallet as a custom network RPC; see services/rpc_firewall.py.
GET  /rpc          Same as /rpc/info (handy when the URL is opened in a browser).
GET  /rpc/info     Network details for adding the RPC to a wallet + firewall thresholds.
GET  /rpc/recent   Recent scored transactions (addresses masked) and verdict totals.

JSON-RPC errors are returned with HTTP 200, as Ethereum nodes do.
"""
from __future__ import annotations

import asyncio
import json
from typing import Any, Dict, List, Literal, Optional

from fastapi import APIRouter, Query, Request, Response
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from inference_engine.api.dependencies import get_app_state
from inference_engine.core.config import settings
from inference_engine.core.rate_limit import client_key
from inference_engine.services import rpc_firewall
from inference_engine.services.counterparty import flagged_addresses
from inference_engine.services.prediction import threshold_for
from inference_engine.services.rpc_firewall import INVALID_REQUEST, PARSE_ERROR, rpc_error

MAX_BODY_BYTES = 2 * 1024 * 1024
MAX_BATCH_SIZE = 100

router = APIRouter(tags=["Wallet RPC"])


class RpcInfoResponse(BaseModel):
    """What a wallet needs to add the firewall as a network, plus its policy."""

    rpc_path: str = Field(default="/rpc", description="Path of the JSON-RPC endpoint on this server.")
    chain_id: int
    chain_id_hex: str
    network_name: str
    currency_symbol: str
    explorer_url: str
    upstream_host: str = Field(description="Host of the node calls are forwarded to.")
    model_loaded: bool
    warn_threshold: Optional[float] = Field(
        description="DeFiGuard probability from which a forwarded transaction is flagged (null if no model)."
    )
    block_threshold: float = Field(description="DeFiGuard probability from which the model alone refuses a transaction.")
    model_blocks: bool = Field(description="False when the model only warns (block_threshold = 1).")
    flagged_addresses: int = Field(description="Size of the counterparty denylist.")
    override_window_seconds: int = Field(description="Re-send window that overrides a block (0 = disabled).")


class CounterpartyFinding(BaseModel):
    severity: Literal["block", "warn"]
    code: str
    message: str


class InterceptionItem(BaseModel):
    id: int
    time: str
    method: str
    verdict: Literal["allowed", "warned", "blocked", "overridden"]
    fraud_probability: float
    warn_threshold: float
    block_threshold: float
    from_address: str
    to_address: str
    value_eth: float
    tx_hash: Optional[str] = None
    upstream_error: Optional[str] = None
    counterparty: List[CounterpartyFinding] = []
    model_signals: List[str] = []
    blocked_by: Optional[Literal["counterparty", "model"]] = None


class RecentInterceptionsResponse(BaseModel):
    items: List[InterceptionItem]
    totals: Dict[str, int]


def _info() -> RpcInfoResponse:
    state = get_app_state()
    return RpcInfoResponse(
        chain_id=settings.rpc_chain_id,
        chain_id_hex=hex(settings.rpc_chain_id),
        network_name=settings.rpc_network_name,
        currency_symbol=settings.rpc_currency_symbol,
        explorer_url=settings.rpc_explorer_url,
        upstream_host=rpc_firewall.upstream_host(),
        model_loaded=state.defiguard_loaded,
        warn_threshold=threshold_for(state.defiguard_model, "pending") if state.defiguard_loaded else None,
        block_threshold=settings.rpc_block_threshold,
        model_blocks=settings.rpc_block_threshold < 1.0,
        flagged_addresses=len(flagged_addresses(settings.flagged_addresses_path)),
        override_window_seconds=settings.rpc_override_window_seconds,
    )


@router.post("/rpc", summary="Ethereum JSON-RPC endpoint with the DeFiGuard firewall")
async def rpc(request: Request) -> Response:
    """Point a wallet here. Transaction submissions are scored by DeFiGuard and
    blocked, flagged or forwarded; all other methods go straight to the upstream node."""
    body = await request.body()
    if len(body) > MAX_BODY_BYTES:
        return JSONResponse(rpc_error(None, INVALID_REQUEST, "Request body too large."), status_code=413)
    try:
        payload: Any = json.loads(body)
    except (ValueError, UnicodeDecodeError):
        return JSONResponse(rpc_error(None, PARSE_ERROR, "Parse error: body is not valid JSON."))

    client = client_key(request)
    state = get_app_state()
    model = state.defiguard_model if state.defiguard_loaded else None

    if isinstance(payload, list):
        if not payload or len(payload) > MAX_BATCH_SIZE:
            return JSONResponse(rpc_error(None, INVALID_REQUEST, f"Batch must hold 1–{MAX_BATCH_SIZE} calls."))
        results = await asyncio.gather(*(rpc_firewall.handle_call(c, client, model) for c in payload))
        responses = [r for r in results if r is not None]
        return JSONResponse(responses) if responses else Response(status_code=204)

    response = await rpc_firewall.handle_call(payload, client, model)
    return JSONResponse(response) if response is not None else Response(status_code=204)


@router.get("/rpc", response_model=RpcInfoResponse, summary="Network details (same as /rpc/info)")
@router.get("/rpc/info", response_model=RpcInfoResponse, summary="Network details and firewall policy")
async def rpc_info() -> RpcInfoResponse:
    return _info()


@router.get("/rpc/recent", response_model=RecentInterceptionsResponse,
            summary="Recently scored wallet transactions (addresses masked)")
async def rpc_recent(limit: int = Query(default=20, ge=1, le=rpc_firewall.LOG_SIZE)) -> RecentInterceptionsResponse:
    return RecentInterceptionsResponse(**rpc_firewall.interception_log.snapshot(limit))
