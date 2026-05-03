from __future__ import annotations

import importlib
from contextvars import ContextVar

from decent_bench.utils.types import SupportedDevices, SupportedFrameworks

from ._abstracts._backend import _Backend
from ._no_backend import _NoBackend

_BACKEND_REGISTRY: dict[SupportedFrameworks, type[_Backend]] = {}
_BACKEND_INSTANCES: dict[SupportedFrameworks, _Backend] = {}
_ACTIVE_BACKEND: ContextVar[SupportedFrameworks | None] = ContextVar("decent_bench.iop2.active_backend", default=None)
_BACKEND: _Backend = _NoBackend()


def set_backend(
    backend: SupportedFrameworks | str,
    device: SupportedDevices | str = SupportedDevices.CPU,
) -> None:
    """
    Set the active backend (and target device) for the current execution context.

    The first call binds both the backend and the device; subsequent calls must use the
    same backend *and* the same device or a :class:`RuntimeError` is raised. This
    single-backend, single-device invariant lets the rest of the interoperability layer
    skip framework dispatch and isinstance checks, and lets backends construct array
    creation routines bound to a specific accelerator.

    Backend modules are auto-imported on demand: the first call to ``set_backend("pytorch")``
    triggers import of ``decent_bench.utils.interoperability_2._pytorch``, whose ``__init__``
    is expected to register the backend via :func:`register_backend`.

    Args:
        backend: A :class:`SupportedFrameworks` value, its canonical string (e.g.
            ``"numpy"``, ``"pytorch"``), or any alias declared by the backend at
            registration time. Aliases are only resolvable after the backend module has
            been loaded; pass the canonical name on the first call to trigger autoload.
        device: Target accelerator. Accepts a :class:`SupportedDevices` value or its
            string equivalent (``"cpu"``, ``"gpu"``, ``"mps"``). Defaults to CPU. The
            backend's array-creation methods produce arrays on this device by default.

    Note:
        Raises :class:`ImportError` if the backend module cannot be imported (e.g. due to
        a missing optional dependency); the failure originates from :func:`_auto_import`.

    Raises:
        RuntimeError: If a different backend (or the same backend with a different device)
            is already active in this context.

    """
    requested = _normalize(backend)
    requested_device = device if isinstance(device, SupportedDevices) else SupportedDevices(device)

    current = _ACTIVE_BACKEND.get()
    if current is not None and current != requested:
        raise RuntimeError(
            f"Backend already set to '{current.value}', cannot set to '{requested.value}'. "
            "A single execution context may only use one backend."
        )

    cached = _instantiate(requested, requested_device)
    if cached.device != requested_device:
        raise RuntimeError(
            f"Backend '{requested.value}' already configured with device "
            f"'{cached.device.value}', cannot reconfigure to '{requested_device.value}'."
        )

    if current is None:
        _ACTIVE_BACKEND.set(requested)
        global _BACKEND  # noqa: PLW0603
        _BACKEND = cached


def register_backend(
    backend: SupportedFrameworks,
    cls: type[_Backend],
) -> None:
    """
    Register a backend class under a :class:`SupportedFrameworks` value.

    Called once per backend module *after* the class definition (rather than as a
    class decorator). Decorator-based registration would mark the decorated class as
    non-extension under mypyc, blocking native compiled-to-compiled dispatch on
    ``_BACKEND.add(...)`` and friends — the call-form keeps concrete backends as
    extension classes.

    Backends are instantiated lazily on first use. Re-registering replaces the
    previous class and discards any cached instance, but keeps existing aliases
    (which still point to the same canonical name).

    Args:
        backend: Canonical backend identifier.
        cls: A concrete subclass of :class:`_Backend`.

    Raises:
        TypeError: If ``cls`` is not a subclass of :class:`_Backend`.

    """
    if not issubclass(cls, _Backend):
        raise TypeError(f"Registered backend must be a subclass of _Backend, got {cls}")
    _BACKEND_REGISTRY[backend] = cls
    _BACKEND_INSTANCES.pop(backend, None)


def reset_backends() -> None:
    """
    Clear the active backend and all cached instances for the current context.

    Intended for tests or tightly scoped execution; not part of normal use. Registry
    entries (classes and aliases) are preserved.
    """
    global _BACKEND  # noqa: PLW0603
    _ACTIVE_BACKEND.set(None)
    _BACKEND_INSTANCES.clear()
    _BACKEND = _NoBackend()


def _normalize(backend: SupportedFrameworks | str) -> SupportedFrameworks:
    if isinstance(backend, SupportedFrameworks):
        return backend
    try:
        return SupportedFrameworks(backend)
    except ValueError as exc:
        valid = ", ".join(f.value for f in SupportedFrameworks)
        raise KeyError(f"Unknown backend '{backend}'. Valid backends: {valid}.") from exc


def _instantiate(backend: SupportedFrameworks, device: SupportedDevices) -> _Backend:
    if backend in _BACKEND_INSTANCES:
        return _BACKEND_INSTANCES[backend]

    if backend not in _BACKEND_REGISTRY:
        _auto_import(backend)

    cls = _BACKEND_REGISTRY.get(backend)
    if cls is None:
        raise KeyError(
            f"Backend '{backend.value}' is not registered. Ensure the corresponding backend module is importable."
        )

    instance = cls(device=device)
    _BACKEND_INSTANCES[backend] = instance
    return instance


def _auto_import(backend: SupportedFrameworks) -> None:
    """
    Import the backend's package so its registration side-effect runs.

    Raises:
        ImportError: If the backend module cannot be imported.

    """
    current_module = __name__.rsplit(".", 1)[0]
    module_name = current_module + f"._{backend.value}"
    try:
        importlib.import_module(module_name)
    except ImportError as exc:
        raise ImportError(
            f"Failed to import the backend module for '{backend.value}'. Ensure the "
            "corresponding backend package is installed and importable."
        ) from exc
