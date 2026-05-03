"""
Interoperability layer (v2).

Single-active-backend variant of the original interoperability package. Each framework
implements :class:`_Backend` and registers itself via :func:`register_backend`. Users
(or the cost-function machinery) bind a backend once with :func:`set_backend`; from
that point on every call in this package routes to that backend without isinstance
dispatch.

Typical usage::

    import decent_bench.utils.interoperability_2 as iop

    iop.set_backend("numpy")
    a = iop.zeros((3, 3))
    iop.set_seed(42)
    s = iop.normal(shape=(2,))

"""

from ._backend_manager import (
    set_backend,
)
from ._decorators import autodecorate_cost_method
from ._iop import (
    absolute,
    add,
    argmax,
    argmin,
    astype,
    choice,
    copy,
    device_of,
    device_to_native,
    diag,
    div,
    dot,
    eye,
    eye_like,
    get_item,
    get_rng_state,
    get_seed,
    iadd,
    idiv,
    imul,
    ipow,
    isub,
    matmul,
    max,
    maximum,
    mean,
    min,
    mul,
    negative,
    norm,
    normal,
    normal_like,
    ones,
    ones_like,
    pow,
    reshape,
    set_item,
    set_rng_state,
    set_seed,
    shape,
    sign,
    sqrt,
    squeeze,
    stack,
    sub,
    sum,
    transpose,
    uniform,
    uniform_like,
    unsqueeze,
    zeros,
    zeros_like,
)

__all__ = [
    "absolute",
    "add",
    "argmax",
    "argmin",
    "astype",
    "autodecorate_cost_method",
    "choice",
    "copy",
    "device_of",
    "device_to_native",
    "diag",
    "div",
    "dot",
    "eye",
    "eye_like",
    "get_item",
    "get_rng_state",
    "get_seed",
    "iadd",
    "idiv",
    "imul",
    "ipow",
    "isub",
    "matmul",
    "max",
    "maximum",
    "mean",
    "min",
    "mul",
    "negative",
    "norm",
    "normal",
    "normal_like",
    "ones",
    "ones_like",
    "pow",
    "reshape",
    "set_backend",
    "set_item",
    "set_rng_state",
    "set_seed",
    "shape",
    "sign",
    "sqrt",
    "squeeze",
    "stack",
    "sub",
    "sum",
    "transpose",
    "uniform",
    "uniform_like",
    "unsqueeze",
    "zeros",
    "zeros_like",
]
