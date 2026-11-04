# -*- coding: utf-8 -*-
"""Counterparty rules, calldata decoding, and the shipped denylist."""
from __future__ import annotations

import json
import re

import pytest

from inference_engine.services.counterparty import (
    DEFAULT_LIST_PATH, check_counterparties, decode_call, load_flagged_addresses,
)

FLAGGED = frozenset({"0x" + "fa" * 20})
WALLET, CONTRACT = "0x" + "ee" * 20, "0x" + "cc" * 20
MAX_UINT = 2**256 - 1


def word(address):
    return address[2:].rjust(64, "0")


async def code_of(address):
    return address == CONTRACT


async def lookup_fails(address):
    return None


def tx(to=CONTRACT, data="0x"):
    return {"from": "0x" + "11" * 20, "to": to, "value": "0x0", "input": data}


class TestDecodeCall:
    def test_approve(self):
        call = decode_call(tx(data="0x095ea7b3" + word(WALLET) + format(MAX_UINT, "064x")))
        assert call.spender == WALLET and call.allowance == MAX_UINT

    def test_set_approval_for_all_true_and_false(self):
        assert decode_call(tx(data="0xa22cb465" + word(WALLET) + "1".rjust(64, "0"))).approve_all
        assert not decode_call(tx(data="0xa22cb465" + word(WALLET) + "0" * 64)).approve_all

    def test_transfer_from_takes_the_second_address(self):
        data = "0x23b872dd" + word("0x" + "11" * 20) + word(WALLET) + format(5, "064x")
        assert decode_call(tx(data=data)).token_recipient == WALLET

    @pytest.mark.parametrize("data", ["0x", "0x095e", "0x095ea7b3", "0x095ea7b3" + "ff" * 64, "0xzz" * 40])
    def test_short_or_malformed_calldata_yields_no_counterparty(self, data):
        call = decode_call(tx(data=data))
        assert call.spender is None and call.token_recipient is None


@pytest.mark.asyncio
class TestRules:
    async def test_plain_transfer_to_an_unlisted_address_is_clean(self):
        report = await check_counterparties(tx(to=WALLET), code_of, FLAGGED)
        assert report.findings == [] and report.unchecked == []

    async def test_flagged_spender_and_wallet_spender_both_reported(self):
        flagged = "0x" + "fa" * 20
        report = await check_counterparties(
            tx(data="0x095ea7b3" + word(flagged) + format(MAX_UINT, "064x")), code_of, FLAGGED)
        assert {f.code for f in report.findings} == {"flagged_counterparty", "approval_to_wallet"}
        assert report.blocks

    async def test_lookup_failure_is_reported_not_guessed(self):
        report = await check_counterparties(
            tx(data="0x095ea7b3" + word(WALLET) + format(MAX_UINT, "064x")), lookup_fails, FLAGGED)
        assert not report.blocks and report.warns
        assert report.unchecked == ["approval spender contract check"]

    async def test_contract_creation_has_no_counterparty(self):
        report = await check_counterparties({"to": None, "input": "0x6080"}, code_of, FLAGGED)
        assert report.findings == []


class TestShippedList:
    def test_list_is_well_formed(self):
        data = json.loads(DEFAULT_LIST_PATH.read_text())
        addresses = load_flagged_addresses(DEFAULT_LIST_PATH)
        assert len(addresses) == data["count"] == 1763
        assert all(re.fullmatch(r"0x[0-9a-f]{40}", a) for a in addresses)

    @pytest.mark.parametrize("address", [
        "0x0000000000000000000000000000000000000000",  # zero address (mints / burns)
        "0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2",  # WETH
        "0xdac17f958d2ee523a2206206994597c13d831ec7",  # USDT
        "0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48",  # USDC
        "0x7a250d5630b4cf539739df2c5dacb4c659f2488d",  # Uniswap V2 router
        "0x3fc91a3afd70395cd496c647d5a6cc9d4b2b7fad",  # Uniswap universal router
        "0x000000000022d473030f116ddee9f6b43ac78ba3",  # Permit2
        "0x00000000000000adc04c56bf30ac9d3c0aaf14dc",  # Seaport 1.5
    ])
    def test_major_contracts_are_not_listed(self, address):
        assert address not in load_flagged_addresses(DEFAULT_LIST_PATH)

    def test_missing_list_degrades_to_empty(self, tmp_path):
        assert load_flagged_addresses(tmp_path / "missing.json") == frozenset()
