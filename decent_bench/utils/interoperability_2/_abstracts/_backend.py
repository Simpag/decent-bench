from abc import ABC
from collections.abc import Callable
from typing import Any

from ._array_creation import _BackendArrayCreation
from ._array_manipulation import _BackendArrayManipulation
from ._linalg import _BackendLinalg
from ._math import _BackendMath
from ._operators import _BackendOperators
from ._rng import _BackendRng


class _Backend(
    _BackendArrayCreation,
    _BackendArrayManipulation,
    _BackendLinalg,
    _BackendMath,
    _BackendOperators,
    _BackendRng,
    ABC,
):
    """
    Abstract base class for a backend that supports array creation, manipulation, linear algebra, math operations, and random number generation.

    This class serves as a blueprint for implementing specific backends (e.g., NumPy, PyTorch, TensorFlow) that can be used interchangeably in the decent_bench framework.
    """


_BACKENDS: dict[str, _Backend] = {}


# Decorator for registering backends
def register_backend(
    name: str, init_kwargs: dict[str, Any] | None = None
) -> Callable[[type[_Backend]], type[_Backend]]:
    def decorator(cls: type[_Backend]) -> type[_Backend]:
        if not issubclass(cls, _Backend):
            raise TypeError(f"Registered backend must be a subclass of _Backend, got {cls}")
        _BACKENDS[name] = cls(**(init_kwargs or {}))
        return cls

    return decorator
