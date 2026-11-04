#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/build_flagged_addresses.py
====================================
Build the wallet RPC firewall's counterparty denylist from the wallets
labelled **fraudulent** in BCCC-DeFiFraudTrans-2025 (the DeFiGuard research
dataset): one ``<address>.json`` file per wallet under ``Fraud/addresses``.

The script refuses to write a list that overlaps the dataset's legitimate
wallets, so a labelling clash can never block a known-legitimate address.

Usage (from backend/):
    python scripts/build_flagged_addresses.py --dataset-dir /path/to/Dataset \\
        --out inference_engine/resources/flagged_addresses.json
"""
from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path

ADDRESS = re.compile(r"^0x[0-9a-f]{40}$")


def wallet_addresses(directory: Path) -> set[str]:
    """Addresses named by the ``<address>.json`` files anywhere under *directory*."""
    return {
        path.stem.lower() for path in directory.rglob("*.json") if ADDRESS.match(path.stem.lower())
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset-dir", required=True, type=Path, help="DeFiGuard research repository root")
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    fraud = wallet_addresses(args.dataset_dir / "Fraud" / "addresses")
    legit = wallet_addresses(args.dataset_dir / "Legit")
    if not fraud:
        raise SystemExit("No fraud wallet files found under Fraud/addresses.")
    overlap = fraud & legit
    if overlap:
        raise SystemExit(f"{len(overlap)} addresses are labelled both fraud and legitimate, e.g. {min(overlap)}.")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "source": "BCCC-DeFiFraudTrans-2025 (Lashkari et al., 2025) — wallets labelled fraudulent",
        "note": "Ethereum mainnet wallets. Wallet-level labels: listed because the dataset flags the wallet.",
        "generated_at": datetime.now(timezone.utc).date().isoformat(),
        "generated_by": "backend/scripts/build_flagged_addresses.py",
        "legitimate_wallets_checked": len(legit),
        "count": len(fraud),
        "addresses": sorted(fraud),
    }, indent=1) + "\n")
    print(f"Wrote {len(fraud)} flagged addresses to {args.out} (0 overlap with {len(legit)} legitimate wallets).")


if __name__ == "__main__":
    main()
