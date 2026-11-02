"""
defiguard.py — DeFiGuard: Multi-Modal Graph-Enhanced DeFi Fraud Detector

Architecture:
  1. TabularEncoder: MLP with LayerNorm over transaction-local features, plus a
     per-feature tokenizer (one token per feature) used as attention keys
  2. GraphEncoder: GATv2Conv encoder over the anonymous receipt ego-graph,
     returning node embeddings and a mean||max pooled graph embedding
  3. CrossAttentionFusion: the tabular embedding attends over the graph's NODE
     embeddings and the graph embedding attends over the tabular FEATURE
     tokens. (Attending between two single vectors would make the softmax
     weight identically 1, i.e. no attention at all.)
  4. DeFiGuard: full model outputting classification logits + contrastive embeddings

`fusion="concat"` replaces cross-attention by concatenation (ablation).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, global_mean_pool, global_max_pool
from torch_geometric.utils import to_dense_batch


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


class FeatureTokenizer(nn.Module):
    """One embedding token per scalar feature: token_f = x_f · w_f + b_f."""

    def __init__(self, n_features, embed_dim):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(n_features, embed_dim) * embed_dim ** -0.5)
        self.bias = nn.Parameter(torch.zeros(n_features, embed_dim))

    def forward(self, x):
        """(B, F) → (B, F, D)"""
        return x.unsqueeze(-1) * self.weight + self.bias


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

    def __init__(self, node_feature_dim=10, edge_feature_dim=6,
                 hidden_dim=128, output_dim=64, n_heads=4,
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

        # Node-level output (keys/values for cross-attention)
        self.node_out = nn.Linear(hidden_dim, output_dim)

        # Output projection (mean + max pooling → 2×hidden → output)
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim * 2, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU()
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, node_features, node_types, edge_index, edge_features, batch,
                batch_size=None, return_nodes=False):
        """
        Args:
            node_features: (N_total, node_feature_dim)
            node_types: (N_total,) int64 node type indices
            edge_index: (2, E_total) edge indices
            edge_features: (E_total, edge_feature_dim)
            batch: (N_total,) batch assignment vector
        Returns:
            (B, output_dim) graph-level embeddings
            [+ (N_total, output_dim) node embeddings if return_nodes]
        """
        # Combine node features with type embeddings
        type_emb = self.type_embedding(node_types)
        x = torch.cat([node_features, type_emb], dim=1)
        x = self.node_proj(x)

        # Project edge features
        edge_attr = self.edge_proj(edge_features) if edge_features.shape[0] > 0 else None

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
        x_mean = global_mean_pool(x, batch, size=batch_size)
        x_max = global_max_pool(x, batch, size=batch_size)
        out = self.output_proj(torch.cat([x_mean, x_max], dim=1))

        if return_nodes:
            return out, self.node_out(x)
        return out


# ─── Cross-Attention Fusion ──────────────────────────────────────────────────

class CrossAttentionFusion(nn.Module):
    """
    Bidirectional multi-head cross-attention between the two modalities.

      tabular query  → keys/values = node embeddings of the ego-graph
      graph query    → keys/values = per-feature tabular tokens
    """

    def __init__(self, embed_dim=64, n_heads=4, dropout=0.1):
        super().__init__()

        # Tabular attending to Graph nodes
        self.tab_to_graph = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=n_heads,
            dropout=dropout, batch_first=True
        )

        # Graph attending to Tabular feature tokens
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

        self.last_attention = None

    def forward(self, tab_emb, tab_tokens, graph_emb, node_emb, batch):
        """
        Args:
            tab_emb: (B, D) tabular embeddings
            tab_tokens: (B, F, D) per-feature tokens
            graph_emb: (B, D) pooled graph embeddings
            node_emb: (N_total, D) node embeddings
            batch: (N_total,) graph assignment of every node
        Returns:
            (B, D) fused embeddings
        """
        nodes, mask = to_dense_batch(node_emb, batch, batch_size=tab_emb.shape[0])

        tab_attended, node_attn = self.tab_to_graph(
            tab_emb.unsqueeze(1), nodes, nodes, key_padding_mask=~mask
        )
        graph_attended, feat_attn = self.graph_to_tab(
            graph_emb.unsqueeze(1), tab_tokens, tab_tokens
        )
        self.last_attention = {"nodes": node_attn.detach(), "features": feat_attn.detach()}

        # Residual + norm
        tab_out = self.norm1(tab_emb + tab_attended.squeeze(1))
        graph_out = self.norm2(graph_emb + graph_attended.squeeze(1))

        # Concatenate and fuse
        return self.fusion_mlp(torch.cat([tab_out, graph_out], dim=1))


# ─── DeFiGuard Full Model ─────────────────────────────────────────────────

class DeFiGuard(nn.Module):
    """
    DeFiGuard: Multi-Modal Graph-Enhanced DeFi Fraud Detector

    Combines:
      - TabularEncoder (+ FeatureTokenizer) for intrinsic transaction features
      - GraphEncoder for receipt ego-graph structural features
      - CrossAttentionFusion (or concatenation, for ablation)
      - Classification head for fraud detection
      - Projection head for contrastive learning
    """

    def __init__(self, tabular_dim, node_feature_dim=10, edge_feature_dim=6,
                 hidden_dim=128, embed_dim=64, n_heads=4,
                 n_node_types=3, dropout=0.2, fusion="cross_attention"):
        super().__init__()
        assert fusion in ("cross_attention", "concat")
        self.fusion_mode = fusion

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
        if fusion == "cross_attention":
            self.tokenizer = FeatureTokenizer(tabular_dim, embed_dim)
            self.fusion = CrossAttentionFusion(
                embed_dim=embed_dim,
                n_heads=n_heads,
                dropout=dropout
            )
        else:
            self.fusion = nn.Sequential(
                nn.Linear(embed_dim * 2, embed_dim),
                nn.LayerNorm(embed_dim),
                nn.GELU(),
                nn.Dropout(dropout)
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

    def get_fused_embedding(self, tabular_features, node_features, node_types,
                            edge_index, edge_features, batch):
        """Fused embedding before the classification head (also used for analysis)."""
        batch_size = tabular_features.shape[0]
        tab_emb = self.tabular_encoder(tabular_features)
        if self.fusion_mode == "concat":
            graph_emb = self.graph_encoder(
                node_features, node_types, edge_index, edge_features, batch, batch_size=batch_size
            )
            return self.fusion(torch.cat([tab_emb, graph_emb], dim=1))
        graph_emb, node_emb = self.graph_encoder(
            node_features, node_types, edge_index, edge_features, batch,
            batch_size=batch_size, return_nodes=True
        )
        return self.fusion(tab_emb, self.tokenizer(tabular_features),
                           graph_emb, node_emb, batch)

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
        fused = self.get_fused_embedding(
            tabular_features, node_features, node_types, edge_index, edge_features, batch
        )
        logits = self.classifier(fused).squeeze(-1)
        embeddings = self.projection(fused)
        return logits, embeddings


# ─── Standalone Components for Ablation ──────────────────────────────────────

class TabularOnlyModel(nn.Module):
    """MLP-only model for tabular features (ablation baseline)."""

    def __init__(self, input_dim, hidden_dim=128, dropout=0.2):
        super().__init__()
        self.encoder = TabularEncoder(input_dim, hidden_dim, 64, 3, dropout)
        self.classifier = nn.Sequential(
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)
        )
        self.projection = nn.Sequential(
            nn.Linear(64, 64),
            nn.GELU(),
            nn.Linear(64, 32)
        )

    def forward(self, x):
        emb = self.encoder(x)
        logits = self.classifier(emb).squeeze(-1)
        proj = self.projection(emb)
        return logits, proj


class GraphOnlyModel(nn.Module):
    """Graph-only model without tabular fusion (ablation baseline)."""

    def __init__(self, node_feature_dim=10, edge_feature_dim=6,
                 hidden_dim=128, n_heads=4, n_node_types=3, dropout=0.2):
        super().__init__()
        self.encoder = GraphEncoder(
            node_feature_dim, edge_feature_dim, hidden_dim, 64,
            n_heads, n_node_types, dropout
        )
        self.classifier = nn.Sequential(
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)
        )
        self.projection = nn.Sequential(
            nn.Linear(64, 64),
            nn.GELU(),
            nn.Linear(64, 32)
        )

    def forward(self, node_features, node_types, edge_index, edge_features, batch):
        batch_size = int(batch.max()) + 1 if batch.numel() else 0
        emb = self.encoder(node_features, node_types, edge_index, edge_features, batch,
                           batch_size=batch_size)
        logits = self.classifier(emb).squeeze(-1)
        proj = self.projection(emb)
        return logits, proj
