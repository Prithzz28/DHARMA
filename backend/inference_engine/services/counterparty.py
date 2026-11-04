# -*- coding: utf-8 -*-
"""
inference_engine/services/counterparty.py
===========================================
Counterparty checks: *who* a transaction pays or gives permissions to.

DeFiGuard was trained on wallet-level labels, so it scores how much a
transaction resembles ones **sent by** fraud-labelled wallets. A user is
usually harmed by the other side of the transaction instead — a scam
recipient, or a drainer that is granted an approval. These rules look at
that other side:

* **flagged_counterparty** (block) — the recipient, the called contract, an
  approved spender / operator, or a token recipient is a wallet labelled
  fraudulent in BCCC-DeFiFraudTrans-2025 (``resources/flagged_addresses.json``).
* **approval_to_wallet** (block) — an unlimited ``approve`` /
  ``increaseAllowance`` or a ``setApprovalForAll(…, true)`` whose spender is a
  regular wallet with no contract code: a common wallet-drainer pattern.
* **unlimited_approval** / **approve_all_nfts** (warn) — the same grants to a
  contract: often legitimate (DEX routers), but worth confirming.
* **limited_approval_to_wallet** (warn) — a bounded approval to a regular wallet.

Contract-code checks call ``eth_getCode`` on the upstream node through the
injected ``has_code`` callable. If that lookup fails the code-dependent rules
are skipped (reported in ``unchecked``), never guessed.
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Awaitable, Callable, Dict, FrozenSet, List, Literal, Optional, Tuple

from inference_engine.services.explainer import mask_address

logger = logging.getLogger(__name__)

Severity = Literal["block", "warn"]
HasCode = Callable[[str], Awaitable[Optional[bool]]]

DEFAULT_LIST_PATH = Path(__file__).resolve().parents[1] / "resources" / "flagged_addresses.json"

#: Allowances at or above this are "unlimited" (wallets and dapps use 2**256 - 1).
UNLIMITED = 2**255

SEL_APPROVE = "095ea7b3"             # approve(address,uint256)
SEL_INCREASE_ALLOWANCE = "39509351"  # increaseAllowance(address,uint256)
SEL_SET_APPROVAL_FOR_ALL = "a22cb465"  # setApprovalForAll(address,bool)
SEL_TRANSFER = "a9059cbb"            # transfer(address,uint256)
#: transferFrom / safeTransferFrom variants: (from, to, …) — the recipient is the 2nd word.
SEL_TRANSFER_FROM = {
    "23b872dd",  # transferFrom(address,address,uint256)
    "42842e0e",  # safeTransferFrom(address,address,uint256)
    "b88d4fde",  # safeTransferFrom(address,address,uint256,bytes)
    "f242432a",  # safeTransferFrom(address,address,uint256,uint256,bytes) — ERC-1155
    "2eb2c2d6",  # safeBatchTransferFrom(address,address,uint256[],uint256[],bytes)
}


@dataclass(frozen=True)
class Finding:
    severity: Severity
    code: str
    message: str


@dataclass
class CounterpartyReport:
    findings: List[Finding] = field(default_factory=list)
    #: Rules skipped because a contract-code lookup failed.
    unchecked: List[str] = field(default_factory=list)

    @property
    def blocks(self) -> bool:
        return any(f.severity == "block" for f in self.findings)

    @property
    def warns(self) -> bool:
        return any(f.severity == "warn" for f in self.findings)


# --------------------------------------------------------------------------- #
# Flagged-address list
# --------------------------------------------------------------------------- #
_flagged: Optional[FrozenSet[str]] = None
_flagged_path: Optional[Path] = None


def load_flagged_addresses(path: Path) -> FrozenSet[str]:
    """Read the denylist (``{"addresses": [...]}``). A missing or unreadable
    file yields an empty list with a warning — the other rules still run."""
    try:
        data = json.loads(Path(path).read_text())
        addresses = frozenset(a.lower() for a in data["addresses"] if isinstance(a, str))
    except (OSError, ValueError, KeyError, TypeError) as exc:
        logger.warning("Flagged-address list %s unavailable (%s); counterparty denylist is empty.", path, exc)
        return frozenset()
    logger.info("Loaded %d flagged counterparty addresses from %s.", len(addresses), path)
    return addresses


def flagged_addresses(path: Path = DEFAULT_LIST_PATH) -> FrozenSet[str]:
    """The denylist at *path*, loaded once and cached."""
    global _flagged, _flagged_path
    if _flagged is None or _flagged_path != Path(path):
        _flagged, _flagged_path = load_flagged_addresses(path), Path(path)
    return _flagged


def reset_flagged_cache() -> None:
    """Forget the cached denylist so the next use reloads it from disk."""
    global _flagged, _flagged_path
    _flagged, _flagged_path = None, None


# --------------------------------------------------------------------------- #
# Contract-code cache (eth_getCode results change rarely)
# --------------------------------------------------------------------------- #
CODE_CACHE_TTL_S = 600.0
_code_cache: Dict[str, Tuple[float, bool]] = {}


async def cached_has_code(address: str, lookup: HasCode) -> Optional[bool]:
    now = time.monotonic()
    hit = _code_cache.get(address)
    if hit and now - hit[0] < CODE_CACHE_TTL_S:
        return hit[1]
    result = await lookup(address)
    if result is not None:
        if len(_code_cache) > 10_000:
            _code_cache.clear()
        _code_cache[address] = (now, result)
    return result


def reset_code_cache() -> None:
    _code_cache.clear()


# --------------------------------------------------------------------------- #
# Calldata decoding
# --------------------------------------------------------------------------- #
def _words(data: str) -> List[str]:
    """32-byte ABI words after the 4-byte selector (hex, no prefix)."""
    body = data[8:]
    return [body[i:i + 64] for i in range(0, len(body) - 63, 64)]


def _address_word(word: str) -> Optional[str]:
    if len(word) != 64 or word[:24].strip("0"):
        return None  # not a left-padded 20-byte address
    return "0x" + word[24:]


def _norm(address: Optional[str]) -> Optional[str]:
    if not isinstance(address, str):
        return None
    address = address.strip().lower()
    return address if len(address) == 42 and address.startswith("0x") else None


@dataclass
class _Call:
    selector: Optional[str] = None
    spender: Optional[str] = None       # approval spender / operator
    allowance: Optional[int] = None     # approve / increaseAllowance amount
    approve_all: bool = False           # setApprovalForAll(…, true)
    token_recipient: Optional[str] = None


def decode_call(tx: Dict) -> _Call:
    data = tx.get("input") or tx.get("data") or "0x"
    data = data[2:].lower() if isinstance(data, str) and data[:2].lower() == "0x" else ""
    if len(data) < 8:
        return _Call()
    selector, words = data[:8], _words(data)
    call = _Call(selector=selector)
    try:
        if selector in (SEL_APPROVE, SEL_INCREASE_ALLOWANCE) and len(words) >= 2:
            call.spender, call.allowance = _address_word(words[0]), int(words[1], 16)
        elif selector == SEL_SET_APPROVAL_FOR_ALL and len(words) >= 2:
            call.spender, call.approve_all = _address_word(words[0]), int(words[1], 16) != 0
        elif selector == SEL_TRANSFER and words:
            call.token_recipient = _address_word(words[0])
        elif selector in SEL_TRANSFER_FROM and len(words) >= 2:
            call.token_recipient = _address_word(words[1])
    except ValueError:
        return _Call(selector=selector)
    return call


# --------------------------------------------------------------------------- #
# Rules
# --------------------------------------------------------------------------- #
async def check_counterparties(
    tx: Dict, has_code: HasCode, flagged: Optional[FrozenSet[str]] = None
) -> CounterpartyReport:
    """Apply the counterparty rules to one transaction."""
    flagged = flagged_addresses() if flagged is None else flagged
    report = CounterpartyReport()
    call = decode_call(tx)
    to = _norm(tx.get("to"))

    roles = []
    if to:
        roles.append(("recipient" if call.selector is None else "contract called", to))
    if call.spender:
        roles.append(("operator" if call.selector == SEL_SET_APPROVAL_FOR_ALL else "approved spender", call.spender))
    if call.token_recipient:
        roles.append(("token recipient", call.token_recipient))
    for role, address in roles:
        if address in flagged:
            report.findings.append(Finding(
                "block", "flagged_counterparty",
                f"The {role} ({mask_address(address)}) is a wallet labelled fraudulent in the "
                "BCCC-DeFiFraudTrans-2025 dataset.",
            ))

    unlimited = call.allowance is not None and call.allowance >= UNLIMITED
    grants = unlimited or call.approve_all or (call.allowance or 0) > 0
    if call.spender and grants:
        spender_has_code = await cached_has_code(call.spender, has_code)
        what = "control of every NFT in this collection" if call.approve_all else (
            "an unlimited token allowance" if unlimited else "a token allowance")
        spender = mask_address(call.spender)
        if spender_has_code is None:
            report.unchecked.append("approval spender contract check")
            if unlimited or call.approve_all:
                report.findings.append(Finding(
                    "warn", "approve_all_nfts" if call.approve_all else "unlimited_approval",
                    f"Grants {what} to {spender}; could not check whether it is a contract.",
                ))
        elif not spender_has_code and (unlimited or call.approve_all):
            report.findings.append(Finding(
                "block", "approval_to_wallet",
                f"Grants {what} to {spender}, a regular wallet rather than a contract — "
                "a common wallet-drainer pattern.",
            ))
        elif not spender_has_code:
            report.findings.append(Finding(
                "warn", "limited_approval_to_wallet",
                f"Grants {what} to {spender}, a regular wallet rather than a contract.",
            ))
        elif unlimited or call.approve_all:
            report.findings.append(Finding(
                "warn", "approve_all_nfts" if call.approve_all else "unlimited_approval",
                f"Grants {what} to contract {spender}. Fine for a DEX or marketplace you trust; "
                "otherwise approve only the amount you need.",
            ))
    return report
