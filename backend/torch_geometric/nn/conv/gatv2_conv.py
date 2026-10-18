# -*- coding: utf-8 -*-
"""
torch_geometric/nn/conv/gatv2_conv.py  (dedicated module file)
================================================================
Stub for ``torch_geometric.nn.conv.gatv2_conv.GATv2Conv``.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch_geometric.nn.aggr.basic import SumAggregation


class GATv2Conv(nn.Module):
    """GATv2 graph attention convolution stub (weight-receiving only).

    Only declares the sub-parameters and sub-modules that appear in the
    stored DeFiGuard state_dict so that ``load_state_dict`` succeeds:

        att             — attention weight vector
        bias            — output bias
        lin_l           — left-branch projection (with bias)
        lin_r           — right-branch projection (no bias)
        lin_edge        — edge-feature projection (no bias)
        aggr_module     — SumAggregation (stored sub-module)
    """

    def __init__(
        self,
        in_channels: int = 64,
        out_channels: int = 64,
        heads: int = 1,
        edge_dim: int = 64,
        **kwargs,
    ) -> None:
        super().__init__()
        self.att = nn.Parameter(torch.zeros(1, heads, out_channels * 2))
        self.bias = nn.Parameter(torch.zeros(out_channels))
        self.lin_l = nn.Linear(in_channels, out_channels * heads, bias=True)
        self.lin_r = nn.Linear(in_channels, out_channels * heads, bias=False)
        self.lin_edge = nn.Linear(edge_dim, out_channels * heads, bias=False)
        self.aggr_module = SumAggregation()

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:  # pragma: no cover
        return self.lin_l(x) if x.dim() == 2 else x
