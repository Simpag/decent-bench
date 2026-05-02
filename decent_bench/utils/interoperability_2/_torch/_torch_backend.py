"""
PyTorch backend for interoperability_2.

This module is intentionally safe to import even when the `torch` package is not
installed.

If PyTorch is available, importing this module registers a `pytorch` backend via
`register_backend(...)`. If PyTorch is not available, this module is a no-op.

Usage (recommended):

    from decent_bench.utils.interoperability_2 import set_backend

    set_backend("pytorch")

The backend manager will try to import this module on-demand.
"""

from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch
else:
    torch = None
    with contextlib.suppress(ImportError, ModuleNotFoundError):
        import torch as _torch

        torch = _torch
