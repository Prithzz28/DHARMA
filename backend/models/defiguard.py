# -*- coding: utf-8 -*-
"""
models/defiguard.py
====================
DeFiGuard: multi-modal graph-enhanced DeFi fraud detector (PyTorch).

Ported verbatim (minus the ablation baselines) from the DeFiGuard training
repository so that the exported ``defi_guard_model.joblib`` runs the *same*
forward pass it was trained with:

  1. TabularEncoder        — MLP over the 45 scaled tabular features.
  2. GraphEncoder          — 2× GATv2Conv over a per-transaction ego-graph.
  3. CrossAttentionFusion  — bidirectional cross-attention of both embeddings.
  4. DeFiGuard             — classifier + contrastive projection heads.

The class names and module path (``models.defiguard``) are referenced by the
pickle stream inside the .joblib file — do NOT rename or move them.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, global_max_pool, global_mean_pool

# ─── Tabular Encoder ─────────────────────────────────────────────────────────

class TabularEncoder(nn.Module):
    """
    MLP encoder for tabular (intrinsic transaction) features.

    Architecture: Input → [Linear → LayerNorm → GELU → Dropout] × N_layers → Output
    """

    def __init__(self, input_dim, hidden_dim=128, output_dim=64,
                 n_layers=3, dropout=0.2):
        super().__init__()

        layers = []
        dims = [input_dim] + [hidden_dim] * (n_layers - 1) + [output_dim]

        for i in range(len(dims) - 1):
            layers.extend([
                nn.Linear(dims[i], dims[i + 1]),
                nn.LayerNorm(dims[i + 1]),
                nn.GELU(),
                nn.Dropout(dropout)
            ])

        self.encoder = nn.Sequential(*layers)

    def forward(self, x):
        """
        Args:
            x: (B, input_dim) tabular features
        Returns:
            (B, output_dim) encoded features
        """
        return self.encoder(x)


# ─── Graph Encoder ───────────────────────────────────────────────────────────

class GraphEncoder(nn.Module):
    """
    GATv2Conv-based graph encoder for ego-graph structural features.

    Uses node type embeddings to handle heterogeneous node types in a
    homogeneous graph framework (more flexible and efficient than HeteroData
    for small ego-graphs).

    Architecture:
      - Node type embedding (learned)
      - Node feature projection
      - 2× GATv2Conv layers with multi-head attention
      - Global pooling (mean + max concatenation)
    """

    def __init__(self, node_feature_dim=8, edge_feature_dim=2,
                 hidden_dim=64, output_dim=64, n_heads=4,
                 n_node_types=3, dropout=0.2):
        super().__init__()

        self.n_node_types = n_node_types

        # Node type embedding
        self.type_embedding = nn.Embedding(n_node_types, 16)

        # Project node features to hidden dim
        self.node_proj = nn.Sequential(
            nn.Linear(node_feature_dim + 16, hidden_dim),  # +16 for type embedding
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )

        # Edge feature projection
        self.edge_proj = nn.Linear(edge_feature_dim, hidden_dim)

        # GATv2Conv layers
        self.conv1 = GATv2Conv(
            hidden_dim, hidden_dim // n_heads,
            heads=n_heads, edge_dim=hidden_dim,
            dropout=dropout, add_self_loops=True
        )
        self.norm1 = nn.LayerNorm(hidden_dim)

        self.conv2 = GATv2Conv(
            hidden_dim, hidden_dim // n_heads,
            heads=n_heads, edge_dim=hidden_dim,
            dropout=dropout, add_self_loops=True
        )
        self.norm2 = nn.LayerNorm(hidden_dim)

        # Output projection (mean + max pooling → 2×hidden → output)
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim * 2, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU()
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, node_features, node_types, edge_index, edge_features, batch):
        """
        Args:
            node_features: (N_total, node_feature_dim)
            node_types: (N_total,) int64 node type indices
            edge_index: (2, E_total) edge indices
            edge_features: (E_total, edge_feature_dim)
            batch: (N_total,) batch assignment vector
        Returns:
            (B, output_dim) graph-level embeddings
        """
        # Combine node features with type embeddings
        type_emb = self.type_embedding(node_types)
        x = torch.cat([node_features, type_emb], dim=1)
        x = self.node_proj(x)

        # Project edge features
        if edge_features.shape[0] > 0:
            edge_attr = self.edge_proj(edge_features)
        else:
            edge_attr = None

        # GATv2Conv layers with residual connections
        identity = x
        x = self.conv1(x, edge_index, edge_attr=edge_attr)
        x = self.norm1(x)
        x = F.gelu(x)
        x = self.dropout(x) + identity

        identity = x
        x = self.conv2(x, edge_index, edge_attr=edge_attr)
        x = self.norm2(x)
        x = F.gelu(x)
        x = self.dropout(x) + identity

        # Global pooling: mean || max
        x_mean = global_mean_pool(x, batch)
        x_max = global_max_pool(x, batch)
        x_pool = torch.cat([x_mean, x_max], dim=1)

        # Output projection
        out = self.output_proj(x_pool)
        return out


# ─── Cross-Attention Fusion ──────────────────────────────────────────────────

class CrossAttentionFusion(nn.Module):
    """
    Multi-head cross-attention layer that fuses tabular and graph embeddings.

    Tabular embedding attends to graph embedding and vice versa, then
    the results are combined.
    """

    def __init__(self, embed_dim=64, n_heads=4, dropout=0.1):
        super().__init__()

        # Tabular attending to Graph
        self.tab_to_graph = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=n_heads,
            dropout=dropout, batch_first=True
        )

        # Graph attending to Tabular
        self.graph_to_tab = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=n_heads,
            dropout=dropout, batch_first=True
        )

        # Layer norms
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

        # Fusion MLP
        self.fusion_mlp = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

    def forward(self, tab_emb, graph_emb):
        """
        Args:
            tab_emb: (B, D) tabular embeddings
            graph_emb: (B, D) graph embeddings
        Returns:
            (B, D) fused embeddings
        """
        # Reshape for attention: (B, 1, D) — single "token" per modality
        tab = tab_emb.unsqueeze(1)
        graph = graph_emb.unsqueeze(1)

        # Cross-attention
        tab_attended, _ = self.tab_to_graph(tab, graph, graph)
        graph_attended, _ = self.graph_to_tab(graph, tab, tab)

        # Residual + norm
        tab_out = self.norm1(tab + tab_attended).squeeze(1)
        graph_out = self.norm2(graph + graph_attended).squeeze(1)

        # Concatenate and fuse
        fused = torch.cat([tab_out, graph_out], dim=1)
        fused = self.fusion_mlp(fused)

        return fused


# ─── DeFiGuard Full Model ─────────────────────────────────────────────────

class DeFiGuard(nn.Module):
    """
    DeFiGuard: Multi-Modal Graph-Enhanced DeFi Fraud Detector

    Combines:
      - TabularEncoder for intrinsic transaction features
      - GraphEncoder for ego-graph structural features
      - CrossAttentionFusion for multi-modal integration
      - Classification head for fraud detection
      - Projection head for contrastive learning
    """

    def __init__(self, tabular_dim, node_feature_dim=8, edge_feature_dim=2,
                 hidden_dim=128, embed_dim=64, n_heads=4,
                 n_node_types=3, dropout=0.2):
        super().__init__()

        # Encoders
        self.tabular_encoder = TabularEncoder(
            input_dim=tabular_dim,
            hidden_dim=hidden_dim,
            output_dim=embed_dim,
            n_layers=3,
            dropout=dropout
        )

        self.graph_encoder = GraphEncoder(
            node_feature_dim=node_feature_dim,
            edge_feature_dim=edge_feature_dim,
            hidden_dim=hidden_dim,
            output_dim=embed_dim,
            n_heads=n_heads,
            n_node_types=n_node_types,
            dropout=dropout
        )

        # Fusion
        self.fusion = CrossAttentionFusion(
            embed_dim=embed_dim,
            n_heads=n_heads,
            dropout=dropout
        )

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.LayerNorm(embed_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, 1)
        )

        # Projection head for contrastive learning
        self.projection = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim // 2)
        )

    def forward(self, tabular_features, node_features, node_types,
                edge_index, edge_features, batch):
        """
        Args:
            tabular_features: (B, tabular_dim)
            node_features: (N_total, node_feature_dim)
            node_types: (N_total,)
            edge_index: (2, E_total)
            edge_features: (E_total, edge_feature_dim)
            batch: (N_total,) batch assignment
        Returns:
            logits: (B,) classification logits
            embeddings: (B, embed_dim//2) contrastive embeddings
        """
        # Encode modalities
        tab_emb = self.tabular_encoder(tabular_features)
        graph_emb = self.graph_encoder(
            node_features, node_types, edge_index, edge_features, batch
        )

        # Fuse
        fused = self.fusion(tab_emb, graph_emb)

        # Classify
        logits = self.classifier(fused).squeeze(-1)

        # Project for contrastive loss
        embeddings = self.projection(fused)

        return logits, embeddings

    def get_fused_embedding(self, tabular_features, node_features, node_types,
                            edge_index, edge_features, batch):
        """Get fused embeddings without classification head (for analysis)."""
        tab_emb = self.tabular_encoder(tabular_features)
        graph_emb = self.graph_encoder(
            node_features, node_types, edge_index, edge_features, batch
        )
        fused = self.fusion(tab_emb, graph_emb)
        return fused
