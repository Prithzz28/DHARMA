# -*- coding: utf-8 -*-
"""
torch_geometric/inspector.py (package init)
=============================================
Stub for classes in ``torch_geometric.inspector``:

    Inspector   — main class: holds _signature_dict, _cls
    Signature   — dict-like signature container
    Parameter   — individual parameter descriptor

These classes are stored in the GATv2Conv portion of the defi_guard_model.joblib
pickle and must exist at deserialization time.  None of them need to do actual
work — the graph path is bypassed in tabular-only inference mode.
"""
from __future__ import annotations

from typing import Any, Dict


class Parameter:
    """Stub for a single inspected parameter (name + type + default)."""

    def __init__(
        self,
        name: str = "",
        param_type: Any = None,
        has_default: bool = False,
        **kwargs: Any,
    ) -> None:
        self.name = name
        self.param_type = param_type
        self.has_default = has_default

    def __repr__(self) -> str:  # pragma: no cover
        return f"Parameter({self.name!r})"


class Signature:
    """Stub for an inspected function signature (maps param name → Parameter)."""

    def __init__(self, params: Dict[str, Parameter] | None = None, **kwargs: Any) -> None:
        self._params: Dict[str, Parameter] = params or {}

    def __getitem__(self, key: str) -> Parameter:  # pragma: no cover
        return self._params.get(key, Parameter(key))

    def __repr__(self) -> str:  # pragma: no cover
        return f"Signature({list(self._params.keys())})"


class Inspector:
    """Stub for torch_geometric.inspector.Inspector.

    Stores _cls and _signature_dict; both are set by the pickle __setstate__.
    """

    def __init__(self, cls: Any = None, **kwargs: Any) -> None:
        self._cls = cls
        self._signature_dict: Dict[str, Signature] = {}

    def __repr__(self) -> str:  # pragma: no cover
        return f"Inspector({getattr(self._cls, '__name__', str(self._cls))})"
