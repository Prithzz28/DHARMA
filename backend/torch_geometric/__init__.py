# -*- coding: utf-8 -*-
"""
torch_geometric/__init__.py
============================
Minimal ``torch_geometric`` compatibility shim for the Web3 Firewall.

The ``defi_guard_model.joblib`` was serialised with PyTorch Geometric classes
embedded in its pickle stream.  This stub package provides just enough class
definitions so that ``joblib.load()`` can reconstruct the object without
requiring a full PyTorch Geometric installation.

Classes provided:
    torch_geometric.nn.conv.gatv2_conv.GATv2Conv
    torch_geometric.nn.aggr.basic.SumAggregation
    torch_geometric.inspector.Inspector

IMPORTANT: These stubs are **weight-receiving only** — they do not implement
graph convolution logic.  The DeFiGuard model is run in tabular-only mode,
bypassing the graph encoder path.  See ``models/predictor.py`` for the
inference implementation.
"""
