"""
Random-number coordination across backends.

The active backend handles its own RNG, but two extra concerns sit above it:

1. Python's :mod:`random` is often used incidentally and must also be seeded.
2. NumPy's RNG is frequently consulted by other frameworks (e.g. dataset shuffling
   helpers, scikit-learn pre-processing) regardless of the active backend, so its state
   must be tracked and restored alongside the active backend's state.

:class:`_RngCoordinator` owns both concerns. RNG functions exposed by ``_iop`` route
through a process-singleton coordinator.

When the active backend *is* numpy, the coordinator avoids double-seeding to keep the
RNG-state snapshot self-consistent.
"""

from __future__ import annotations

import random
from typing import Any

from decent_bench.utils.array import Array
from decent_bench.utils.array._array import _NoBackendSet
from decent_bench.utils.interoperability_2._abstracts._backend import _Backend
from decent_bench.utils.interoperability_2._backend_manager import _instantiate
from decent_bench.utils.types import SupportedFrameworks

# Force numpy backend registration; the coordinator always needs numpy regardless of
# the active backend. NumPy is a hard dependency of decent-bench, so this is safe.
import decent_bench.utils.interoperability_2._numpy  # noqa: E402, F401

_NUMPY_STATE_KEY = "__numpy_rng_state__"
_PYTHON_RANDOM_KEY = "__python_random_state__"


# Module-level cache for the active backend. Bound by
# :func:`decent_bench.utils.interoperability_2.set_backend`. Reading ``_BACKEND`` on
# the hot RNG path (``normal``/``uniform``/etc.) is one global-name load instead of
# the ContextVar + dict lookup that ``get_backend()`` does.
_BACKEND: _Backend = _NoBackendSet()  # type: ignore[assignment]


def _set_active_backend(backend: _Backend) -> None:
    """Bind the active backend (called from set_backend)."""
    global _BACKEND  # noqa: PLW0603
    _BACKEND = backend


class _RngCoordinator:
    """Coordinate RNG seeding/state across the active backend, NumPy, and Python's random."""

    def __init__(self) -> None:
        self._global_seed: int | None = None

    def set_seed(self, seed: int, *, set_global_seed: bool = True) -> None:
        """
        Seed Python's ``random``, NumPy's RNG, and the active backend's RNG.

        Args:
            seed: Base seed.
            set_global_seed: If False, leaves :func:`get_seed` untouched. Use this for
                trial-local reseeding where the externally observable base seed must be
                preserved.

        """
        random.seed(seed)
        active = _BACKEND
        active.set_seed(seed)
        numpy_backend = self._numpy_backend()
        if numpy_backend is not active:
            numpy_backend.set_seed(seed)
        if set_global_seed:
            self._global_seed = seed

    def get_seed(self) -> int | None:
        """Return the seed last passed to :meth:`set_seed` (with ``set_global_seed=True``)."""
        return self._global_seed

    def get_rng_state(self) -> dict[str, Any]:
        """
        Snapshot the RNG state of the active backend, NumPy (if auxiliary), and Python's random.

        The active backend's state is returned as-is. If the active backend is not NumPy,
        NumPy's state is embedded under the reserved key ``"__numpy_rng_state__"``. The
        Python ``random`` state is always embedded under ``"__python_random_state__"`` so
        that incidental ``random.random()`` calls survive a snapshot/restore round-trip.

        """
        active = _BACKEND
        state = active.get_rng_state()
        state[_PYTHON_RANDOM_KEY] = random.getstate()
        numpy_backend = self._numpy_backend()
        if numpy_backend is not active:
            state[_NUMPY_STATE_KEY] = numpy_backend.get_rng_state()
        return state

    def set_rng_state(self, state: dict[str, Any]) -> None:
        """Restore a snapshot produced by :meth:`get_rng_state`."""
        # Copy so we can mutate without surprising the caller.
        state = dict(state)
        python_state = state.pop(_PYTHON_RANDOM_KEY, None)
        if python_state is not None:
            random.setstate(python_state)
        active = _BACKEND
        numpy_backend = self._numpy_backend()
        if numpy_backend is not active:
            numpy_state = state.pop(_NUMPY_STATE_KEY, None)
            if numpy_state is not None:
                numpy_backend.set_rng_state(numpy_state)
        active.set_rng_state(state)

    def _numpy_backend(self) -> _Backend:
        # Resolved lazily so the coordinator works before numpy is registered/instantiated.
        return _instantiate(SupportedFrameworks.NUMPY)


_COORDINATOR = _RngCoordinator()


def set_seed(seed: int) -> None:
    """Seed Python ``random``, NumPy, and the active backend's RNG with ``seed``."""
    _COORDINATOR.set_seed(seed)


def _set_seed_without_global(seed: int) -> None:
    """
    Seed without changing the value returned by :func:`get_seed`.

    Used for trial-local reseeding where the externally observable base seed must be preserved.
    """
    _COORDINATOR.set_seed(seed, set_global_seed=False)


def get_seed() -> int | None:
    """Return the most recently set global seed, or ``None`` if unset."""
    return _COORDINATOR.get_seed()


def get_rng_state() -> dict[str, Any]:
    """Return a snapshot of the active backend's RNG state."""
    return _COORDINATOR.get_rng_state()


def set_rng_state(state: dict[str, Any]) -> None:
    """Restore an RNG snapshot produced by :func:`get_rng_state`."""
    _COORDINATOR.set_rng_state(state)


def normal(mean: float = 0.0, std: float = 1.0, shape: tuple[int, ...] = ()) -> Array:
    """Draw normally distributed samples on the active backend."""
    return _BACKEND.normal(mean, std, shape)


def uniform(low: float = 0.0, high: float = 1.0, shape: tuple[int, ...] = ()) -> Array:
    """Draw uniformly distributed samples on the active backend."""
    return _BACKEND.uniform(low, high, shape)


def normal_like(array: Array, mean: float = 0.0, std: float = 1.0) -> Array:
    """Draw normally distributed samples shaped like ``array``."""
    return _BACKEND.normal_like(array, mean, std)


def uniform_like(array: Array, low: float = 0.0, high: float = 1.0) -> Array:
    """Draw uniformly distributed samples shaped like ``array``."""
    return _BACKEND.uniform_like(array, low, high)


def choice(array: Array, size: int, replace: bool = True) -> Array:
    """Sample ``size`` elements from ``array``."""
    return _BACKEND.choice(array, size, replace)
