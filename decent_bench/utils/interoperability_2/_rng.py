from __future__ import annotations

import contextlib
import random
from collections.abc import Iterable
from copy import deepcopy
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, cast

import numpy as np

from decent_bench.utils.types import SupportedDevices, SupportedFrameworks

jax = None
jnp = None
tf = None
torch = None

with contextlib.suppress(ImportError, ModuleNotFoundError):
    import torch as _torch

    torch = _torch

with contextlib.suppress(ImportError, ModuleNotFoundError):
    import tensorflow as _tf

    tf = _tf

with contextlib.suppress(ImportError, ModuleNotFoundError):
    import jax.numpy as _jnp

    jnp = _jnp

with contextlib.suppress(ImportError, ModuleNotFoundError):
    import jax as _jax

    jax = _jax

if TYPE_CHECKING:
    from jax import Array as JaxArray
    from tensorflow.random import Generator as TensorflowGenerator
    from torch import Generator as TorchGenerator


@dataclass
class _RngState:
    global_seed: int | None = None
    numpy_rng: np.random.Generator = field(default_factory=np.random.default_rng)
    jax_key: JaxArray | None = None
    tf_generator: TensorflowGenerator | None = None
    torch_generators: dict[SupportedDevices, TorchGenerator] = field(default_factory=dict)


_STATE = _RngState(
    numpy_rng=np.random.default_rng(),
    jax_key=(jax.random.key(random.randint(0, 2**32 - 1)) if jax else None),
    tf_generator=(tf.random.Generator.from_non_deterministic_state() if tf else None),
    torch_generators={},
)


def _selected_frameworks(frameworks: Iterable[SupportedFrameworks] | None) -> set[SupportedFrameworks]:
    if frameworks is None:
        return set(SupportedFrameworks)
    return set(frameworks)


def set_seed(
    seed: int,
    frameworks: Iterable[SupportedFrameworks] | None = None,
) -> None:
    """
    Set random seeds across supported frameworks.

    Args:
        seed: Base seed to use.
        frameworks: Optional subset of frameworks to seed. If ``None``, all are seeded.

    """
    _set_seed(seed=seed, frameworks=frameworks, set_global_seed=True)


def _set_seed(
    seed: int,
    frameworks: Iterable[SupportedFrameworks] | None = None,
    *,
    set_global_seed: bool = True,
) -> None:
    """
    Set random seeds across supported frameworks.

    Args:
        seed: Base seed to use.
        frameworks: Optional subset of frameworks to seed. If ``None``, all are seeded.
        set_global_seed: Whether to update the globally tracked seed returned by
            :func:`get_seed`. Set this to ``False`` for trial-local reseeding where
            preserving the external base seed is required.

    """
    selected = _selected_frameworks(frameworks)

    random.seed(seed)

    if SupportedFrameworks.NUMPY in selected:
        # If a user uses legacy np.random functions
        np.random.seed(seed)  # noqa: NPY002
        _STATE.numpy_rng = np.random.default_rng(seed)

    if torch and SupportedFrameworks.PYTORCH in selected:
        torch.manual_seed(seed)
        _STATE.torch_generators.clear()

    if tf and SupportedFrameworks.TENSORFLOW in selected:
        tf.random.set_seed(seed)
        _STATE.tf_generator = tf.random.Generator.from_seed(seed, alg="philox")

    if jax and SupportedFrameworks.JAX in selected:
        _STATE.jax_key = jax.random.key(seed)

    if set_global_seed:
        _STATE.global_seed = seed


def get_seed() -> int | None:
    """Return the current global seed if one was set explicitly."""
    return _STATE.global_seed


def rng_numpy() -> np.random.Generator:
    """Return the shared NumPy generator used by interoperability random functions."""
    return _STATE.numpy_rng


def rng_jax() -> JaxArray:
    """
    Split and return the next JAX sub-key while advancing global JAX RNG state.

    Raises:
        RuntimeError: if JAX is not installed.

    """
    if not jax or _STATE.jax_key is None:
        raise RuntimeError("JAX is not installed.")
    _STATE.jax_key, sub_key = jax.random.split(_STATE.jax_key)
    return cast("JaxArray", sub_key)


def rng_torch(device: SupportedDevices = SupportedDevices.CPU) -> TorchGenerator:
    """
    Return a torch.Generator for a given device.

    Raises:
        RuntimeError: if PyTorch is not installed.

    """
    if not torch:
        raise RuntimeError("PyTorch is not installed.")

    if device in _STATE.torch_generators:
        return _STATE.torch_generators[device]

    framework_device = device_to_framework_device(device, SupportedFrameworks.PYTORCH)
    generator: TorchGenerator = torch.Generator(device=framework_device)
    if _STATE.global_seed is not None:
        generator.manual_seed(torch.initial_seed())
    _STATE.torch_generators[device] = generator
    return generator


def rng_tensorflow() -> TensorflowGenerator:
    """
    Return a TensorFlow random generator.

    Raises:
        RuntimeError: if TensorFlow is not installed.

    """
    if not tf:
        raise RuntimeError("TensorFlow is not installed.")

    if _STATE.tf_generator is None:
        # Only for type chekcing, in practice _tf_generator should always be initialized if tf is available
        raise RuntimeError("TensorFlow random generator is not initialized.")

    return _STATE.tf_generator


def get_rng_state(frameworks: Iterable[SupportedFrameworks] | None = None) -> dict[str, Any]:
    """
    Return a picklable snapshot of all managed RNG states.

    Args:
        frameworks: Optional subset of frameworks to seed. If ``None``, all are seeded.

    """
    selected = _selected_frameworks(frameworks)

    state: dict[str, Any] = {
        "seed": _STATE.global_seed,
        "python_random_state": random.getstate(),
        "numpy_bit_generator_state": deepcopy(_STATE.numpy_rng.bit_generator.state),
        "numpy_rng_state": np.random.get_state(),  # noqa: NPY002 # Include legacy state for users who use legacy np.random functions
    }

    if torch and SupportedFrameworks.PYTORCH in selected:
        state["torch_rng_state"] = torch.random.get_rng_state()
        if torch.cuda.is_available():
            state["torch_cuda_rng_state"] = torch.cuda.get_rng_state_all()
        state["torch_generators"] = {device: gen.get_state() for device, gen in _STATE.torch_generators.items()}

    if tf and _STATE.tf_generator is not None and SupportedFrameworks.TENSORFLOW in selected:
        state["tf_generator_state"] = _STATE.tf_generator.state.numpy()

    if jax and _STATE.jax_key is not None and SupportedFrameworks.JAX in selected:
        state["jax_key"] = jax.random.key_data(_STATE.jax_key)

    return state


def set_rng_state(state: dict[str, Any]) -> None:
    """Restore a RNG snapshot created by ``get_rng_state``."""
    if "seed" in state:
        _STATE.global_seed = state["seed"]

    if "python_random_state" in state:
        random.setstate(state["python_random_state"])

    if "numpy_bit_generator_state" in state:
        _STATE.numpy_rng = np.random.default_rng()
        _STATE.numpy_rng.bit_generator.state = state["numpy_bit_generator_state"]

    if "numpy_rng_state" in state:
        np.random.set_state(state["numpy_rng_state"])  # noqa: NPY002

    if torch and "torch_rng_state" in state:
        torch.random.set_rng_state(state["torch_rng_state"])

    if torch and "torch_cuda_rng_state" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["torch_cuda_rng_state"])

    if torch and "torch_generators" in state:
        _STATE.torch_generators.clear()
        for device, generator_state in state["torch_generators"].items():
            framework_device = device_to_framework_device(device, SupportedFrameworks.PYTORCH)
            generator = torch.Generator(device=framework_device)
            generator.set_state(generator_state)
            _STATE.torch_generators[device] = generator

    if tf and "tf_generator_state" in state:
        _STATE.tf_generator = tf.random.Generator.from_state(state["tf_generator_state"], alg="philox")

    if jax and "jax_key" in state:
        _STATE.jax_key = jax.random.wrap_key_data(state["jax_key"])
