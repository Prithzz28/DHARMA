# -*- coding: utf-8 -*-
"""
inference_engine/core/config.py
================================
Centralised settings for the inference engine, loaded from environment
variables and ``.env`` files via Pydantic Settings.

``.env`` lookup does not depend on the working directory: the repository-root
``.env`` is read first, then ``backend/.env`` (later files win). Real
environment variables override both.

Environment variables
---------------------
DEFIGUARD_MODEL_PATH      DeFiGuard .joblib (relative paths resolve against backend/).
FRAUD_THRESHOLD           Optional override (0, 1) of the per-mode block thresholds that
                          ship inside the model (chosen on validation data in training).
RATE_LIMIT_PER_MINUTE     Scoring requests allowed per client IP per minute (0 disables).
HOST / PORT               Uvicorn bind address.
LOG_LEVEL                 Python logging level (DEBUG, INFO, WARNING …).
CORS_ORIGINS              Comma-separated allowed origins ("*" for development).
GROQ_API_KEY              Optional — enables LLM risk explanations.
GROQ_MODEL                Groq chat model for explanations (default openai/gpt-oss-120b).
LLM_MAX_CALLS_PER_MINUTE  Process-wide cap on Groq calls (0 disables the LLM).

Wallet RPC firewall (POST /rpc)
UPSTREAM_RPC_URL              Ethereum node allowed calls are forwarded to (default: public Sepolia).
RPC_CHAIN_ID / RPC_NETWORK_NAME / RPC_CURRENCY_SYMBOL / RPC_EXPLORER_URL
                              Network details shown to users when they add the RPC to a wallet.
RPC_BLOCK_THRESHOLD           DeFiGuard probability from which the model alone refuses a transaction
                              (default 1.0 = model only warns; counterparty checks block).
FLAGGED_ADDRESSES_PATH        Counterparty denylist (default: BCCC fraud-labelled wallets).
RPC_OVERRIDE_WINDOW_SECONDS   Re-sending a blocked transaction within this window forwards it (0 disables).
RPC_RATE_LIMIT_PER_MINUTE     Non-transaction RPC calls allowed per client IP per minute (0 disables).
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional

from dotenv import dotenv_values
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND_DIR = Path(__file__).resolve().parents[2]
REPO_ROOT = BACKEND_DIR.parent
ENV_FILES = (REPO_ROOT / ".env", BACKEND_DIR / ".env")

#: Settings from the former Random Forest / ensemble setup. They no longer have
#: any effect; ``obsolete_settings_in_use`` lets startup warn about them.
OBSOLETE_SETTINGS = ("MODEL_PATH", "ENSEMBLE_MODE", "RF_WEIGHT", "DG_WEIGHT")


class Settings(BaseSettings):
    """Application-wide configuration loaded from the environment."""

    model_config = SettingsConfigDict(
        env_file=ENV_FILES,
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ------------------------------------------------------------------ #
    # Model
    # ------------------------------------------------------------------ #
    defiguard_model_path: Path = Path("models/defi_guard_model.joblib")

    #: Optional block-threshold override. By default each mode (pending /
    #: receipt) uses the threshold stored in the model, which was selected on
    #: address-disjoint validation data during training (see README).
    fraud_threshold: Optional[float] = Field(default=None, gt=0.0, lt=1.0)

    # ------------------------------------------------------------------ #
    # Server
    # ------------------------------------------------------------------ #
    host: str = "0.0.0.0"
    port: int = 8001
    log_level: str = "INFO"
    cors_origins: str = "*"
    rate_limit_per_minute: int = Field(default=60, ge=0)

    # ------------------------------------------------------------------ #
    # Risk explanation (LLM)
    # ------------------------------------------------------------------ #
    groq_api_key: str = ""
    #: Groq retires models over time; check https://console.groq.com/docs/models.
    groq_model: str = "openai/gpt-oss-120b"
    llm_max_calls_per_minute: int = Field(default=20, ge=0)

    # ------------------------------------------------------------------ #
    # Wallet RPC firewall (POST /rpc)
    # ------------------------------------------------------------------ #
    #: Node that allowed transactions and all other RPC calls are forwarded
    #: to. It may embed a provider key — it is never shown to clients.
    upstream_rpc_url: str = "https://ethereum-sepolia-rpc.publicnode.com"
    rpc_chain_id: int = Field(default=11155111, gt=0)
    rpc_network_name: str = "Sepolia"
    rpc_currency_symbol: str = "SepoliaETH"
    rpc_explorer_url: str = "https://sepolia.etherscan.io"
    #: DeFiGuard probability from which the model alone refuses a transaction.
    #: 1.0 (default) = the model only warns; blocks come from counterparty checks.
    #: DeFiGuard's labels are wallet-level (it scores resemblance to transactions
    #: *sent by* fraud-labelled wallets), so it does not justify blocking a sender.
    rpc_block_threshold: float = Field(default=1.0, gt=0.0, le=1.0)
    #: Counterparty denylist (JSON {"addresses": [...]}); relative paths resolve against backend/.
    flagged_addresses_path: Path = Path("inference_engine/resources/flagged_addresses.json")
    #: Re-sending a blocked transaction within this many seconds forwards it (0 disables).
    rpc_override_window_seconds: int = Field(default=120, ge=0)
    #: Non-transaction RPC calls (balances, blocks, gas …) allowed per client IP per minute.
    rpc_rate_limit_per_minute: int = Field(default=600, ge=0)

    @field_validator("defiguard_model_path", "flagged_addresses_path")
    @classmethod
    def _resolve_against_backend(cls, value: Path) -> Path:
        return value if value.is_absolute() else (BACKEND_DIR / value).resolve()


def obsolete_settings_in_use() -> List[str]:
    """Names of removed settings still present in the environment or .env files."""
    found = {name for name in OBSOLETE_SETTINGS if name in os.environ}
    for env_file in ENV_FILES:
        if env_file.exists():
            keys = {key.upper() for key in dotenv_values(env_file)}
            found.update(name for name in OBSOLETE_SETTINGS if name in keys)
    return sorted(found)


#: Module-level singleton — import this everywhere instead of re-creating.
settings = Settings()
