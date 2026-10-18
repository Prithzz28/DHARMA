# -*- coding: utf-8 -*-
"""
models/defiguard.py
====================
PyTorch ``nn.Module`` stub definitions for the DeFiGuard model.

These classes **must** exist at deserialization time so that joblib/pickle can
reconstruct the DeFiGuard model object from the ``.joblib`` file.

Architecture (reverse-engineered from the stored state_dict and pickle stream):

    TabularEncoder   — 3-block dense encoder over tabular feature inputs:
                       Linear → BatchNorm → GELU → Dropout  (×3 blocks)
                       State-dict keys: tabular_encoder.encoder.{0,1,4,5,8,9}.{weight,bias}

    GraphEncoder     — GATConv-based graph encoder:
                       type_embedding, node_proj, edge_proj,
                       conv1, norm1, conv2, norm2, output_proj, dropout
                       State-dict keys: graph_encoder.*

    CrossAttentionFusion — Cross-attention fusion of tabular + graph streams:
                       tab_to_graph, graph_to_tab, norm1, norm2, fusion_mlp
                       State-dict keys: fusion.*

    DeFiGuard        — Full DeFi Guard neural network:
                       tabular_encoder (TabularEncoder) + _model (DeFiGuard shell)
                       classifier + projection heads
                       State-dict keys: classifier.*, projection.*

Note: ALL classes below are **imported by pickle** during
``joblib.load("defi_guard_model.joblib")``.  Do NOT rename, move, or delete them.
"""
from __future__ import annotations

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# TabularEncoder — dense feature encoder
# ---------------------------------------------------------------------------

class TabularEncoder(nn.Module):
    """Encoder sub-network for tabular feature inputs.

    3-block sequential encoder:
        Linear → BatchNorm1d → GELU → Dropout   ×3
    """

    def __init__(self) -> None:
        super().__init__()
        self.encoder: nn.Sequential = nn.Sequential()

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover
        return self.encoder(x)


# ---------------------------------------------------------------------------
# GraphEncoder — GATConv-based graph encoder
# ---------------------------------------------------------------------------

class GraphEncoder(nn.Module):
    """GATConv-based graph encoder (stub for deserialization).

    Sub-modules (match state_dict key prefixes exactly):
        type_embedding      — nn.Embedding
        node_proj           — Sequential: Linear → LayerNorm → GELU
        edge_proj           — nn.Linear
        conv1, conv2        — GATConv-compatible stub layers (with edge features)
        norm1, norm2        — nn.LayerNorm
        output_proj         — Sequential: Linear → LayerNorm → GELU
        dropout             — nn.Dropout

    Note: In tabular-only inference mode this module is loaded but not called.
    """

    def __init__(self) -> None:
        super().__init__()
        # Placeholders — exact shapes restored from stored state_dict
        self.type_embedding: nn.Embedding = nn.Embedding(10, 64)
        self.node_proj: nn.Sequential = nn.Sequential(
            nn.Linear(64, 64),
            nn.LayerNorm(64),
            nn.GELU(),
        )
        self.edge_proj: nn.Linear = nn.Linear(8, 64)
        self.conv1: _GATConvStub = _GATConvStub(64, 64)
        self.norm1: nn.LayerNorm = nn.LayerNorm(64)
        self.conv2: _GATConvStub = _GATConvStub(64, 64)
        self.norm2: nn.LayerNorm = nn.LayerNorm(64)
        self.output_proj: nn.Sequential = nn.Sequential(
            nn.Linear(64, 64),
            nn.LayerNorm(64),
            nn.GELU(),
        )
        self.dropout: nn.Dropout = nn.Dropout(0.3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover
        return x


# ---------------------------------------------------------------------------
# CrossAttentionFusion — fusion of tabular + graph streams
# ---------------------------------------------------------------------------

class CrossAttentionFusion(nn.Module):
    """Cross-attention fusion block for combining tabular and graph embeddings.

    Sub-modules (match state_dict key prefixes exactly):
        tab_to_graph    — MultiheadAttention (tabular query → graph KV)
        graph_to_tab    — MultiheadAttention (graph query → tabular KV)
        norm1, norm2    — LayerNorm
        fusion_mlp      — Sequential MLP
    """

    def __init__(self, embed_dim: int = 64) -> None:
        super().__init__()
        self.tab_to_graph: nn.MultiheadAttention = nn.MultiheadAttention(
            embed_dim, num_heads=4, batch_first=True
        )
        self.graph_to_tab: nn.MultiheadAttention = nn.MultiheadAttention(
            embed_dim, num_heads=4, batch_first=True
        )
        self.norm1: nn.LayerNorm = nn.LayerNorm(embed_dim)
        self.norm2: nn.LayerNorm = nn.LayerNorm(embed_dim)
        self.fusion_mlp: nn.Sequential = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(0.3),
        )

    def forward(  # pragma: no cover
        self,
        tab: torch.Tensor,
        graph: torch.Tensor,
    ) -> torch.Tensor:
        return torch.cat([tab, graph], dim=-1)


# ---------------------------------------------------------------------------
# DeFiGuard — top-level model shell stored inside DeFiGuardPredictor
# ---------------------------------------------------------------------------

class DeFiGuard(nn.Module):
    """Full DeFi Guard neural network (tabular encoder + graph encoder + fusion + classifier).

    This is the inner ``_model`` stored inside a ``DeFiGuardPredictor`` wrapper.
    Sub-modules:
        tabular_encoder     — TabularEncoder
        graph_encoder       — GraphEncoder
        fusion              — CrossAttentionFusion
        classifier          — Sequential MLP head → sigmoid
        projection          — Optional contrastive projection head
    """

    def __init__(self) -> None:
        super().__init__()
        self.tabular_encoder: TabularEncoder = TabularEncoder()
        self.graph_encoder: GraphEncoder = GraphEncoder()
        self.fusion: CrossAttentionFusion = CrossAttentionFusion()
        self.classifier: nn.Sequential = nn.Sequential()
        self.projection: nn.Sequential = nn.Sequential()

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover
        tab_emb = self.tabular_encoder(x)
        return self.classifier(tab_emb)


# ---------------------------------------------------------------------------
# Internal stubs for GATConv-like layers
# ---------------------------------------------------------------------------

class _GATConvStub(nn.Module):
    """Minimal stub for a GATConv-compatible layer (weight-receiving only)."""

    def __init__(self, in_channels: int = 64, out_channels: int = 64) -> None:
        super().__init__()
        self.lin_l = nn.Linear(in_channels, out_channels, bias=True)
        self.lin_r = nn.Linear(in_channels, out_channels, bias=False)
        self.lin_edge = nn.Linear(in_channels, out_channels, bias=False)
        self.att = nn.Parameter(torch.zeros(1, 1, out_channels * 2))
        self.bias = nn.Parameter(torch.zeros(out_channels))
        self.aggr_module = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover
        return self.lin_l(x)
