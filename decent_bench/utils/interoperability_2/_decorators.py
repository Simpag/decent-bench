from __future__ import annotations

from collections.abc import Callable
from typing import Any


def autodecorate_cost_method[T: Callable[..., Any]](superclass_method: T) -> Callable[[Callable[..., Any]], T]:
    """
    Decorate Cost methods to automatically convert :class:`~decent_bench.utils.array.Array` args and return types.

    With single-active-backend semantics, this decorator no longer needs to dispatch on
    framework: it can unwrap :class:`~decent_bench.utils.array.Array` arguments to the
    backend-native type, call the cost method, and wrap any returned native array back
    in :class:`~decent_bench.utils.array.Array`.

    Args:
        superclass_method: The method from the superclass (e.g. ``Cost.function``) that
            is being overridden.

    Note:
        Implementation deferred until the new structure stabilizes.

    """
    raise NotImplementedError("autodecorate_cost_method has not been ported to interoperability_2 yet.")
