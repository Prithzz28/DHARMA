# -*- coding: utf-8 -*-
"""
models/__init__.py
==================
Package init for the ``models`` namespace required by defi_guard_model.joblib.

The DeFiGuard model (.joblib) was serialised with custom classes under the
``models`` package.  This package exposes ``DeFiGuardPredictor`` (from
``models.predictor``) and ``DeFiGuard`` (from ``models.defiguard``) so that
``joblib.load()`` can successfully deserialise the stored object.
"""
