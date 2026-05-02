from __future__ import annotations

from abc import ABC

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
    Abstract base class for a backend.

    The backend supports array creation, manipulation, linear algebra, math operations,
    operators, and random number generation.
    """
