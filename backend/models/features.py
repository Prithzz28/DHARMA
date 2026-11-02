"""
features.py — Leakage-Free Transaction Features and Receipt Ego-Graphs

Single source of truth for turning ONE Ethereum transaction (JSON-RPC
transaction object + optional receipt) into DeFiGuard model inputs. The same
code is used to build the training set and at inference time (the Mini Pro
firewall ships a byte-identical copy), so there is no train/serve skew.

Design rules (each closes a leakage path found in the v1 pipeline):
  1. Transaction-local only — nothing about the sender's or recipient's history
     (no per-address aggregates, no token holdings, no account age).
  2. No absolute-time identifiers — block number, transaction index, cumulative
     block gas and signature/EIP type fields encode *when* a transaction was
     mined, which differs between the fraud and legitimate collections.
  3. No address identity — addresses become anonymous graph nodes described
     only by their role in this transaction; no hashes of addresses.
  4. No fitted vocabularies — function selectors and event topics are fixed
     a priori from the ERC-20/721 and common DEX interfaces, not learned from
     the labelled data.

Two feature groups:
  PRE_EXEC_FEATURES  — computable from the signed transaction alone
                       (mempool / wallet "pending" mode).
  RECEIPT_FEATURES   — require the execution receipt, obtained after inclusion
                       or by simulating the pending transaction.

Receipt ego-graph (receipt mode):
  Nodes:  TX (type 0) · ACCOUNT (type 1) · TOKEN/EMITTER CONTRACT (type 2)
  Edges:  sender→TX (submit) · TX→recipient (call) · TX→emitter (emit) ·
          party→token→party (token transfer, two hops) · plus reverse edges.
"""

import math

import numpy as np


# ─── Constants ───────────────────────────────────────────────────────────────

ZERO_ADDRESS = "0x" + "0" * 40

# Event topics (keccak256 of the event signature) — fixed by the token standards
TOPIC_TRANSFER = "0xddf252ad1be2c89b69c2b068fc378daa952ba7f163c4a11628f55a4df523b3ef"
TOPIC_APPROVAL = "0x8c5be1e5ebec7d5bd14f71427d1e84f3dd0314c0f7b2291e5b200ac8c7c3b925"
TOPIC_APPROVAL_FOR_ALL = "0x17307eab39ab6107e8899845ad3d59bd9653f200f220920489ca2b5937696c31"

# 4-byte function selectors, grouped by intent — fixed from public interfaces
SELECTOR_GROUPS = {
    "sel_token_transfer": {"0xa9059cbb"},                          # transfer
    "sel_approval": {"0x095ea7b3", "0xa22cb465", "0x39509351"},    # approve, setApprovalForAll, increaseAllowance
    "sel_transfer_from": {"0x23b872dd", "0x42842e0e", "0xb88d4fde"},  # transferFrom, safeTransferFrom(×2)
    "sel_dex_swap": {
        "0x7ff36ab5", "0x18cbafe5", "0x38ed1739", "0xfb3bdb41",    # Uniswap V2 router swaps
        "0x8803dbee", "0x4a25d94a", "0x5c11d795", "0xb6f9de95",
        "0x791ac947", "0x414bf389", "0xc04b8d59", "0xdb3e2198",    # Uniswap V3 router
        "0x04e45aaf", "0xb858183f", "0x3593564c", "0x24856bc3",    # SwapRouter02 / Universal Router
    },
    "sel_multicall": {"0xac9650d8", "0x5ae401dc", "0x1f0464d1", "0x252dba42"},
    "sel_weth": {"0xd0e30db0", "0x2e1a7d4d"},                      # deposit, withdraw
}
KNOWN_SELECTORS = set().union(*SELECTOR_GROUPS.values())

PRE_EXEC_FEATURES = [
    "log_value_eth",
    "log_gas_limit",
    "log_gas_price_gwei",
    "log_nonce",
    "is_contract_creation",
    "is_self_transfer",
    "has_calldata",
    "log_calldata_bytes",
    "calldata_zero_frac",
    *SELECTOR_GROUPS.keys(),
    "sel_unknown",
]

RECEIPT_FEATURES = [
    "status_failed",
    "log_gas_used",
    "gas_used_ratio",
    "log_n_logs",
    "log_n_emitters",
    "log_n_transfers",
    "log_n_nft_transfers",
    "log_n_approvals",
    "log_n_transfer_tokens",
    "log_n_transfer_parties",
    "sender_sends_tokens",
    "sender_receives_tokens",
    "has_mint",
    "has_burn",
    "created_contract",
]

ALL_FEATURES = PRE_EXEC_FEATURES + RECEIPT_FEATURES

# Graph schema
NODE_TX, NODE_ACCOUNT, NODE_TOKEN = 0, 1, 2
N_NODE_TYPES = 3
NODE_FEATURES = [
    "is_sender", "is_recipient", "is_emitter", "is_created", "is_zero_address",
    "log_in_degree", "log_out_degree", "log_transfers_in", "log_transfers_out",
    "log_logs_emitted",
]
EDGE_FEATURES = ["is_submit", "is_call", "is_emit", "is_transfer", "log_amount", "is_reverse"]
NODE_FEATURE_DIM = len(NODE_FEATURES)
EDGE_FEATURE_DIM = len(EDGE_FEATURES)
MAX_GRAPH_NODES = 64


# ─── Parsing Helpers ─────────────────────────────────────────────────────────

def hex_to_int(value, default=0):
    """Parse a JSON-RPC quantity ('0x1a'), int, float or decimal string."""
    if value is None:
        return default
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return int(value) if math.isfinite(value) else default
    try:
        s = str(value).strip()
        return int(s, 16) if s.lower().startswith("0x") else int(float(s))
    except (ValueError, TypeError, OverflowError):
        return default


def norm_address(value):
    """Lower-case 0x address, or '' when absent."""
    if not value or not isinstance(value, str):
        return ""
    return value.strip().lower()


def topic_to_address(topic):
    """Last 20 bytes of a 32-byte indexed topic, as a 0x address."""
    if not isinstance(topic, str) or len(topic) < 42:
        return ""
    return "0x" + topic[-40:].lower()


def _log1p(x):
    return float(math.log1p(max(float(x), 0.0)))


def split_rpc_result(raw):
    """Split a stored eth_getTransactionReceipt/eth_getTransactionByHash pair.

    Accepts either the dataset's {"result": [receipt, tx, ...]} files or an
    already separated dict. Returns (tx, receipt); either may be None.
    """
    items = raw.get("result", raw) if isinstance(raw, dict) else raw
    if isinstance(items, dict):
        items = [items]
    tx, receipt = None, None
    for item in items or []:
        if not isinstance(item, dict):
            continue
        if "logs" in item and "gasUsed" in item:
            receipt = item
        elif "input" in item or "nonce" in item or "gas" in item:
            tx = item
    return tx, receipt


# ─── Tabular Features ────────────────────────────────────────────────────────

def pre_exec_features(tx):
    """Features computable from the signed (pending) transaction only."""
    tx = tx or {}
    value_wei = hex_to_int(tx.get("value"))
    gas_limit = hex_to_int(tx.get("gas"))
    gas_price = hex_to_int(tx.get("gasPrice")) or hex_to_int(tx.get("maxFeePerGas"))
    nonce = hex_to_int(tx.get("nonce"))
    sender = norm_address(tx.get("from"))
    recipient = norm_address(tx.get("to"))

    data = tx.get("input") or tx.get("data") or "0x"
    data = data[2:] if isinstance(data, str) and data.startswith("0x") else ""
    n_bytes = len(data) // 2
    zero_bytes = sum(1 for i in range(0, n_bytes * 2, 2) if data[i:i + 2] == "00")
    selector = ("0x" + data[:8].lower()) if n_bytes >= 4 else None

    feats = {
        "log_value_eth": _log1p(value_wei / 1e18),
        "log_gas_limit": _log1p(gas_limit),
        "log_gas_price_gwei": _log1p(gas_price / 1e9),
        "log_nonce": _log1p(nonce),
        "is_contract_creation": float(not recipient),
        "is_self_transfer": float(bool(sender) and sender == recipient),
        "has_calldata": float(n_bytes > 0),
        "log_calldata_bytes": _log1p(n_bytes),
        "calldata_zero_frac": zero_bytes / n_bytes if n_bytes else 0.0,
    }
    for group, selectors in SELECTOR_GROUPS.items():
        feats[group] = float(selector in selectors)
    feats["sel_unknown"] = float(selector is not None and selector not in KNOWN_SELECTORS)
    return feats


def _decode_logs(receipt):
    """Decode receipt logs into emitters, token transfers and approvals."""
    emitters, transfers = [], []
    n_logs = n_nft = n_approvals = 0
    logs = (receipt or {}).get("logs")
    for log in logs if isinstance(logs, list) else []:
        if not isinstance(log, dict):
            continue
        n_logs += 1
        emitter = norm_address(log.get("address"))
        if emitter:
            emitters.append(emitter)
        topics = log.get("topics")
        topics = topics if isinstance(topics, list) else []
        topic0 = topics[0].lower() if topics and isinstance(topics[0], str) else ""
        if topic0 == TOPIC_TRANSFER and len(topics) >= 3:
            is_nft = len(topics) >= 4
            n_nft += int(is_nft)
            amount = 1 if is_nft else hex_to_int(log.get("data"), default=0)
            transfers.append((emitter, topic_to_address(topics[1]), topic_to_address(topics[2]), amount))
        elif topic0 in (TOPIC_APPROVAL, TOPIC_APPROVAL_FOR_ALL):
            n_approvals += 1
    return emitters, transfers, n_logs, n_nft, n_approvals


def receipt_features(tx, receipt):
    """Features that need the execution receipt (or a simulation of it)."""
    tx, receipt = tx or {}, receipt or {}
    sender = norm_address(receipt.get("from") or tx.get("from"))
    gas_limit = hex_to_int(tx.get("gas"))
    gas_used = hex_to_int(receipt.get("gasUsed"))
    emitters, transfers, n_logs, n_nft, n_approvals = _decode_logs(receipt)
    parties = {a for _, f, t, _ in transfers for a in (f, t)} - {ZERO_ADDRESS, ""}

    return {
        "status_failed": float(bool(receipt) and hex_to_int(receipt.get("status"), 1) == 0),
        "log_gas_used": _log1p(gas_used),
        "gas_used_ratio": min(gas_used / gas_limit, 1.0) if gas_limit else 0.0,
        "log_n_logs": _log1p(n_logs),
        "log_n_emitters": _log1p(len(set(emitters))),
        "log_n_transfers": _log1p(len(transfers)),
        "log_n_nft_transfers": _log1p(n_nft),
        "log_n_approvals": _log1p(n_approvals),
        "log_n_transfer_tokens": _log1p(len({e for e, _, _, _ in transfers})),
        "log_n_transfer_parties": _log1p(len(parties)),
        "sender_sends_tokens": float(any(f == sender for _, f, _, _ in transfers)),
        "sender_receives_tokens": float(any(t == sender for _, _, t, _ in transfers)),
        "has_mint": float(any(f == ZERO_ADDRESS for _, f, _, _ in transfers)),
        "has_burn": float(any(t == ZERO_ADDRESS for _, _, t, _ in transfers)),
        "created_contract": float(bool(norm_address(receipt.get("contractAddress")))),
    }


def tabular_vector(tx, receipt=None, feature_names=ALL_FEATURES):
    """Ordered float32 feature vector; receipt features are 0 without a receipt."""
    feats = pre_exec_features(tx)
    feats.update(receipt_features(tx, receipt))
    return np.array([feats[name] for name in feature_names], dtype=np.float32)


# ─── Receipt Ego-Graph ───────────────────────────────────────────────────────

def build_ego_graph(tx, receipt=None, max_nodes=MAX_GRAPH_NODES):
    """Anonymous, transaction-local ego-graph.

    Without a receipt the graph is just sender → TX → recipient (pending mode).
    Node identity never enters the features: an address is described only by
    its roles and degrees inside this one transaction.
    """
    tx, receipt = tx or {}, receipt or {}
    sender = norm_address(receipt.get("from") or tx.get("from"))
    recipient = norm_address(receipt.get("to") or tx.get("to"))
    created = norm_address(receipt.get("contractAddress"))
    emitters, transfers, _, _, _ = _decode_logs(receipt)

    index, types, roles = {}, [NODE_TX], [dict()]
    raw_edges = []  # (src, dst, kind, log_amount)

    def node(addr, node_type):
        if addr in index:
            if node_type == NODE_TOKEN:
                types[index[addr]] = NODE_TOKEN
            return index[addr]
        if len(types) >= max_nodes:
            return None
        index[addr] = len(types)
        types.append(node_type)
        roles.append(dict())
        return index[addr]

    value_eth = hex_to_int(tx.get("value")) / 1e18
    if sender:
        s = node(sender, NODE_ACCOUNT)
        roles[s]["is_sender"] = 1.0
        raw_edges.append((s, 0, "submit", 0.0))
    callee = recipient or created
    if callee:
        c = node(callee, NODE_ACCOUNT)
        if c is not None:
            roles[c]["is_recipient" if recipient else "is_created"] = 1.0
            raw_edges.append((0, c, "call", _log1p(value_eth)))
    if created and created != callee:
        c = node(created, NODE_ACCOUNT)
        if c is not None:
            roles[c]["is_created"] = 1.0
            raw_edges.append((0, c, "call", 0.0))
    for emitter in emitters:
        e = node(emitter, NODE_TOKEN)
        if e is None:
            continue
        roles[e]["n_logs"] = roles[e].get("n_logs", 0) + 1
        if roles[e]["n_logs"] == 1:
            roles[e]["is_emitter"] = 1.0
            raw_edges.append((0, e, "emit", 0.0))
    for token, frm, to, amount in transfers:
        t, f, d = node(token, NODE_TOKEN), node(frm, NODE_ACCOUNT), node(to, NODE_ACCOUNT)
        if None in (t, f, d):
            continue
        log_amount = math.log10(1.0 + amount) / 10.0
        raw_edges.append((f, t, "transfer", log_amount))
        raw_edges.append((t, d, "transfer", log_amount))
        roles[f]["transfers_out"] = roles[f].get("transfers_out", 0) + 1
        roles[d]["transfers_in"] = roles[d].get("transfers_in", 0) + 1
    for addr, i in index.items():
        if addr == ZERO_ADDRESS:
            roles[i]["is_zero_address"] = 1.0

    n = len(types)
    kinds = ("submit", "call", "emit", "transfer")
    edge_index, edge_feat = [], []
    in_deg, out_deg = np.zeros(n), np.zeros(n)
    for src, dst, kind, log_amount in raw_edges:
        onehot = [float(kind == k) for k in kinds]
        edge_index.append((src, dst))
        edge_feat.append(onehot + [log_amount, 0.0])
        edge_index.append((dst, src))
        edge_feat.append(onehot + [log_amount, 1.0])
        out_deg[src] += 1
        in_deg[dst] += 1

    node_feat = np.zeros((n, NODE_FEATURE_DIM), dtype=np.float32)
    for i, r in enumerate(roles):
        node_feat[i] = [
            r.get("is_sender", 0.0), r.get("is_recipient", 0.0), r.get("is_emitter", 0.0),
            r.get("is_created", 0.0), r.get("is_zero_address", 0.0),
            _log1p(in_deg[i]), _log1p(out_deg[i]),
            _log1p(r.get("transfers_in", 0)), _log1p(r.get("transfers_out", 0)),
            _log1p(r.get("n_logs", 0)),
        ]

    return {
        "node_features": node_feat,
        "node_types": np.array(types, dtype=np.int64),
        "edge_index": (np.array(edge_index, dtype=np.int64).T if edge_index
                       else np.zeros((2, 0), dtype=np.int64)),
        "edge_features": (np.array(edge_feat, dtype=np.float32) if edge_feat
                          else np.zeros((0, EDGE_FEATURE_DIM), dtype=np.float32)),
    }


def graph_summary(graph):
    """Hand-crafted graph statistics (for the tree-ensemble + graph baseline)."""
    nt, nf, ef = graph["node_types"], graph["node_features"], graph["edge_features"]
    n_edges = ef.shape[0] // 2
    return np.array([
        math.log1p(len(nt)),
        math.log1p(n_edges),
        math.log1p(int((nt == NODE_TOKEN).sum())),
        math.log1p(int((nt == NODE_ACCOUNT).sum())),
        float(nf[:, 5].max()) if len(nf) else 0.0,  # max log in-degree
        float(nf[:, 6].max()) if len(nf) else 0.0,  # max log out-degree
        math.log1p(float(ef[:, 3].sum()) / 2) if n_edges else 0.0,  # transfer hops
    ], dtype=np.float32)


GRAPH_SUMMARY_FEATURES = [
    "g_log_nodes", "g_log_edges", "g_log_token_nodes", "g_log_account_nodes",
    "g_max_log_in_degree", "g_max_log_out_degree", "g_log_transfer_hops",
]
