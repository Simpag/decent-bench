"""
Public re-export of the :class:`Array` wrapper.

The implementation lives in :mod:`._array` so the mypyc-compiled artifacts
(``_array.cpython-*.so`` and ``_array__mypyc.cpython-*.so``) stay contained inside
this package directory rather than scattered across ``decent_bench/utils/``.
"""

from ._array import Array

__all__ = ["Array"]
