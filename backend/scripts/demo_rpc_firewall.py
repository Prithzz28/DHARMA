#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/demo_rpc_firewall.py
==============================
Send one signed Sepolia transaction per firewall path through ``POST /rpc``
and print each verdict — fills the dashboard's live feed for a demo.

  1. plain 0.01 ETH transfer from a fresh wallet  → warned  (sender pattern)
  2. ETH transfer to a fraud-labelled wallet       → blocked (counterparty)
  3. unlimited WETH approval to a plain wallet     → blocked (drainer pattern)
  4. unlimited WETH approval to a DEX router       → warned  (approval to a contract)

By default a throwaway, unfunded key signs, so the Sepolia node rejects the
forwarded ones ("insufficient funds") and nothing reaches the chain. Pass
``--private-key`` with a funded *testnet* key to broadcast the forwarded ones.

Usage (from backend/, with the API running):
    python scripts/demo_rpc_firewall.py [--rpc http://localhost:8001/rpc] [--private-key 0x...]
"""
from __future__ import annotations

import argparse
import json
import urllib.request

from eth_account import Account
from eth_utils import to_checksum_address

from inference_engine.services.counterparty import DEFAULT_LIST_PATH

SEPOLIA = 11155111
WETH = "0xfFf9976782d46CC05630D1f6eBAb18b2324d6B14"      # Sepolia WETH
ROUTER = "0x3bFA4769FB09eefC5a80d6E87c3B9C650f7Ae48E"    # Uniswap SwapRouter02 on Sepolia
RECIPIENT = "0x8c5d7e12f3a4b5c6d7890abcdef3456789012cd3"
MAX_UINT = 2**256 - 1


def rpc(url: str, method: str, params: list) -> dict:
    body = json.dumps({"jsonrpc": "2.0", "id": 1, "method": method, "params": params}).encode()
    request = urllib.request.Request(url, body, {"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=30) as resp:
        return json.loads(resp.read())


def approve(spender: str) -> str:
    return "0x095ea7b3" + spender[2:].lower().rjust(64, "0") + format(MAX_UINT, "064x")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rpc", default="http://localhost:8001/rpc")
    parser.add_argument("--private-key", help="funded Sepolia key (default: a new, unfunded one)")
    args = parser.parse_args()

    account = Account.from_key(args.private_key) if args.private_key else Account.create()
    flagged = json.loads(DEFAULT_LIST_PATH.read_text())["addresses"][0]
    plain_wallet = Account.create().address
    nonce = int(rpc(args.rpc, "eth_getTransactionCount", [account.address, "pending"])["result"], 16)
    fees = {"maxFeePerGas": 30 * 10**9, "maxPriorityFeePerGas": 2 * 10**9, "chainId": SEPOLIA}

    scenarios = [
        ("Plain transfer from this wallet", RECIPIENT, 10**16, "0x", 21_000),
        ("Transfer to a fraud-labelled wallet", flagged, 10**16, "0x", 21_000),
        ("Unlimited WETH approval to a plain wallet", WETH, 0, approve(plain_wallet), 60_000),
        ("Unlimited WETH approval to a DEX router", WETH, 0, approve(ROUTER), 60_000),
    ]
    print(f"Sender {account.address} (nonce {nonce}) → {args.rpc}\n")
    for title, to, value, data, gas in scenarios:
        signed = account.sign_transaction({"to": to_checksum_address(to), "value": value, "data": data,
                                           "gas": gas, "nonce": nonce, **fees})
        reply = rpc(args.rpc, "eth_sendRawTransaction", ["0x" + bytes(signed.raw_transaction).hex()])
        if "result" in reply:
            nonce += 1  # broadcast — the next transaction needs the next nonce
            print(f"• {title}\n    forwarded, tx {reply['result']}\n")
        else:
            print(f"• {title}\n    {reply['error']['message']}\n")
    print("Verdicts and reasons: GET /rpc/recent, or the dashboard's live feed.")


if __name__ == "__main__":
    main()
