# -*- coding: utf-8 -*-
"""
inference_engine/services/raw_transaction.py
==============================================
Decodes a signed raw transaction — the hex string a wallet passes to
``eth_sendRawTransaction`` — into the JSON-RPC transaction object DeFiGuard
scores (``from``, ``to``, ``value``, ``gas``, ``gasPrice``/``maxFeePerGas``,
``nonce``, ``input``, ``chainId``).

Supported envelopes (EIP-2718): legacy (incl. EIP-155), type 1 (EIP-2930),
type 2 (EIP-1559), type 3 (EIP-4844, canonical or network form) and type 4
(EIP-7702). The sender is recovered from the signature; if recovery fails the
transaction is still decoded with ``from`` set to ``None``.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import rlp
from eth_account import Account
from eth_utils import keccak

logger = logging.getLogger(__name__)

#: Largest raw transaction accepted (bytes). Blob transactions in network form
#: carry up to six 128 KiB blobs.
MAX_RAW_TX_BYTES = 1024 * 1024

_DYNAMIC_FEE = ("chainId", "nonce", "maxPriorityFeePerGas", "maxFeePerGas", "gas", "to", "value", "input")

#: Leading RLP fields per transaction type (signature and access lists follow).
_LAYOUTS = {
    0: ("nonce", "gasPrice", "gas", "to", "value", "input"),
    1: ("chainId", "nonce", "gasPrice", "gas", "to", "value", "input"),
    2: _DYNAMIC_FEE,
    3: _DYNAMIC_FEE,
    4: _DYNAMIC_FEE,
}
#: Every signed envelope ends with three signature fields (v/yParity, r, s).
_SIGNATURE_FIELDS = 3


class RawTransactionError(ValueError):
    """The raw transaction is not a well-formed signed Ethereum transaction."""


def _int(value: bytes) -> int:
    return int.from_bytes(value, "big")


def _recover_sender(raw: bytes) -> Optional[str]:
    try:
        return Account.recover_transaction(raw).lower()
    except Exception as exc:  # noqa: BLE001 — any failure just means "sender unknown"
        logger.info("Could not recover the transaction sender (%s: %s).", type(exc).__name__, exc)
        return None


def decode_raw_transaction(raw_hex: Any) -> Dict[str, Any]:
    """Decode a signed raw transaction into a JSON-RPC transaction dict.

    Raises ``RawTransactionError`` if *raw_hex* is not a signed transaction.
    """
    if not isinstance(raw_hex, str):
        raise RawTransactionError("Raw transaction must be a hex string.")
    digits = raw_hex[2:] if raw_hex[:2].lower() == "0x" else raw_hex
    if len(digits) > 2 * MAX_RAW_TX_BYTES:
        raise RawTransactionError(f"Raw transaction exceeds {MAX_RAW_TX_BYTES} bytes.")
    try:
        raw = bytes.fromhex(digits)
    except ValueError as exc:
        raise RawTransactionError("Raw transaction is not valid hex.") from exc
    if not raw:
        raise RawTransactionError("Raw transaction is empty.")

    if raw[0] >= 0xC0:
        tx_type, payload = 0, raw
    elif raw[0] in _LAYOUTS:
        tx_type, payload = raw[0], raw[1:]
    else:
        raise RawTransactionError(f"Unsupported transaction type 0x{raw[0]:02x}.")

    try:
        items = rlp.decode(payload)
    except rlp.exceptions.DecodingError as exc:
        raise RawTransactionError("Raw transaction is not valid RLP.") from exc

    canonical = raw
    if tx_type == 3 and isinstance(items, list) and items and isinstance(items[0], list):
        # Network form: [tx_payload_body, blobs, commitments, proofs].
        items = items[0]
        canonical = bytes([3]) + rlp.encode(items)

    layout = _LAYOUTS[tx_type]
    if (
        not isinstance(items, list)
        or len(items) < len(layout) + _SIGNATURE_FIELDS
        or not all(isinstance(item, bytes) for item in items[: len(layout)])
    ):
        raise RawTransactionError("Raw transaction has the wrong structure for its type.")

    fields = dict(zip(layout, items))
    to = fields["to"]
    if len(to) not in (0, 20):
        raise RawTransactionError("Raw transaction has an invalid 'to' address.")

    tx: Dict[str, Any] = {"type": hex(tx_type)}
    for name, value in fields.items():
        if name == "to":
            tx["to"] = "0x" + value.hex() if value else None
        elif name == "input":
            tx["input"] = "0x" + value.hex()
        else:
            tx[name] = hex(_int(value))
    if tx_type == 0:
        v = _int(items[len(layout)]) if isinstance(items[len(layout)], bytes) else 0
        if v >= 35:  # EIP-155: v = chainId * 2 + 35/36
            tx["chainId"] = hex((v - 35) // 2)

    tx["from"] = _recover_sender(canonical)
    tx["hash"] = "0x" + keccak(canonical).hex()
    return tx
