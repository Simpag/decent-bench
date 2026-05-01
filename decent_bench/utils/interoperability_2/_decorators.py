from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypeVar

T = TypeVar("T", bound=Callable[..., Any])
"""A generic callable type variable."""


def autodecorate_cost_method[T: Callable[..., Any]](superclass_method: T) -> Callable[[Callable[..., Any]], T]:
    """
    Decorate Cost methods to automatically convert :class:`~decent_bench.utils.array.Array` args and return types.

    It automatically converts input :class:`~decent_bench.utils.array.Array` arguments
    to the cost's framework-specific array type and wraps the output based on the
    superclass method's return type annotation.

    Args:
        superclass_method: The method from the superclass (e.g., `Cost.function`) that is being overridden.

    Note:
        * Only arguments that are instances of :class:`~decent_bench.utils.array.Array` are converted.
            Other types are passed through unchanged.
        * The first input argument of the decorated function must be ``x``.
            This is to determine the target array type for output conversion. Otherwise a :class:`ValueError` is raised.
        * Emits a warning if an input array's framework differs from the cost's framework.
            This may lead to unexpected behavior or performance issues.

    """
    raise NotImplementedError("This function is not yet implemented.")
