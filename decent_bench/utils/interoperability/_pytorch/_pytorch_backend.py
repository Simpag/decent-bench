"""
PyTorch backend for interoperability_2.

Importing this module registers the backend via :func:`register_backend`, so the
package can be auto-loaded on the first ``set_backend("pytorch")`` call.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray

from decent_bench.utils.array import Array
from decent_bench.utils.interoperability._abstracts._backend import _Backend
from decent_bench.utils.interoperability._backend_manager import register_backend
from decent_bench.utils.types import ArrayKey, SupportedDevices, SupportedFrameworks


def _unwrap(array: Any) -> Any:  # noqa: ANN401
    """Return the underlying value of an :class:`Array`, or pass ``array`` through."""
    return array.value if isinstance(array, Array) else array


class PyTorchBackend(_Backend):  # noqa: PLR0904
    """PyTorch implementation of :class:`_Backend`."""

    def __init__(self, device: SupportedDevices = SupportedDevices.CPU) -> None:
        super().__init__(device)
        self._native_device: str = self.device_to_native(device)
        self._generator: torch.Generator = torch.Generator(device=self._native_device)

    # Array creation

    def zeros(self, shape: tuple[int, ...]) -> Array:
        return Array(torch.zeros(shape, device=self._native_device))

    def zeros_like(self, array: Array) -> Array:
        return Array(torch.zeros_like(array.value))

    def ones(self, shape: tuple[int, ...]) -> Array:
        return Array(torch.ones(shape, device=self._native_device))

    def ones_like(self, array: Array) -> Array:
        return Array(torch.ones_like(array.value))

    def eye(self, n: int) -> Array:
        return Array(torch.eye(n, device=self._native_device))

    def eye_like(self, array: Array) -> Array:
        v = array.value
        return Array(torch.eye(*v.shape[-2:], dtype=v.dtype, device=v.device))

    def device_to_native(self, device: SupportedDevices) -> str:
        if device == SupportedDevices.CPU:
            return "cpu"
        if device == SupportedDevices.GPU:
            return "cuda"
        if device == SupportedDevices.MPS:
            return "mps"
        raise ValueError(f"Unsupported device: {device}")

    def device_of(self, array: Array) -> SupportedDevices:
        kind = array.value.device.type
        if kind == "cpu":
            return SupportedDevices.CPU
        if kind == "cuda":
            return SupportedDevices.GPU
        if kind == "mps":
            return SupportedDevices.MPS
        raise TypeError(f"Unsupported PyTorch device type: {kind}")

    # Array manipulation

    def copy(self, array: Array) -> Array:
        return Array(array.value.detach().clone())

    def to_numpy(self, array: Array) -> NDArray[Any]:
        """Return the value of an :class:`Array` as a NumPy array."""
        v = array.value
        if isinstance(v, torch.Tensor):
            return v.cpu().numpy()
        return np.asarray(v)

    def from_numpy(self, array: NDArray[Any]) -> Array:
        return Array(torch.from_numpy(array).to(device=self._native_device))

    def to_array(self, array: float | bool) -> Array:
        return Array(torch.tensor(array, device=self._native_device))

    def stack(self, arrays: Sequence[Array], dim: int = 0) -> Array:
        if len(arrays) == 0:
            raise ValueError("Cannot stack an empty sequence of arrays.")
        return Array(torch.stack([a.value for a in arrays], dim=dim))

    def reshape(self, array: Array, shape: tuple[int, ...]) -> Array:
        return Array(torch.reshape(array.value, shape))

    def transpose(self, array: Array, dim: tuple[int, ...] | None = None) -> Array:
        v = array.value
        dims = dim if dim is not None else tuple(reversed(range(v.ndim)))
        return Array(torch.permute(v, dims=dims))

    def shape(self, array: Array) -> tuple[int, ...]:
        return tuple(array.value.shape)

    def squeeze(self, array: Array, dim: int | tuple[int, ...] | None = None) -> Array:
        v = array.value
        if dim is None:
            return Array(torch.squeeze(v))
        return Array(torch.squeeze(v, dim=dim))

    def unsqueeze(self, array: Array, dim: int) -> Array:
        return Array(torch.unsqueeze(array.value, dim=dim))

    def diag(self, array: Array) -> Array:
        return Array(torch.diag(array.value))

    def astype(self, array: Array, dtype: type[float | int | bool]) -> float | int | bool:
        return dtype(array.value.item())

    # Linalg

    def dot(self, array1: Array, array2: Array) -> Array:
        return Array(torch.dot(array1.value, array2.value))

    def matmul(self, array1: Array, array2: Array) -> Array:
        return Array(array1.value @ array2.value)

    def norm(
        self,
        array: Array,
        p: float = 2,
        dim: int | tuple[int, ...] | None = None,
        keepdims: bool = False,
    ) -> Array:
        return Array(torch.linalg.norm(array.value, ord=p, dim=dim, keepdim=keepdims))

    # Math reductions

    def sum(self, array: Array, dim: int | tuple[int, ...] | None = None, keepdims: bool = False) -> Array:
        v = array.value
        if dim is None:
            return Array(torch.sum(v))
        return Array(torch.sum(v, dim=dim, keepdim=keepdims))

    def mean(self, array: Array, dim: int | tuple[int, ...] | None = None, keepdims: bool = False) -> Array:
        v = array.value
        if dim is None:
            return Array(torch.mean(v))
        return Array(torch.mean(v, dim=dim, keepdim=keepdims))

    def min(self, array: Array, dim: int | tuple[int, ...] | None = None, keepdims: bool = False) -> Array:
        v = array.value
        if dim is None:
            return Array(torch.min(v))
        return Array(torch.amin(v, dim=dim, keepdim=keepdims))

    def max(self, array: Array, dim: int | tuple[int, ...] | None = None, keepdims: bool = False) -> Array:
        v = array.value
        if dim is None:
            return Array(torch.max(v))
        return Array(torch.amax(v, dim=dim, keepdim=keepdims))

    # Math elementwise — operands may be Array or scalar (operator dunders pass either).
    # ``Array | float`` covers both: PEP 484's numeric tower implicitly admits ``int``.

    def add(self, array1: Array | float, array2: Array | float) -> Array:
        return Array(torch.add(_unwrap(array1), _unwrap(array2)))

    def iadd[T: Array](self, array1: T, array2: Array | float) -> T:
        array1.value.add_(_unwrap(array2))
        return array1

    def sub(self, array1: Array | float, array2: Array | float) -> Array:
        return Array(torch.sub(_unwrap(array1), _unwrap(array2)))

    def isub[T: Array](self, array1: T, array2: Array | float) -> T:
        array1.value.sub_(_unwrap(array2))
        return array1

    def mul(self, array1: Array | float, array2: Array | float) -> Array:
        return Array(torch.mul(_unwrap(array1), _unwrap(array2)))

    def imul[T: Array](self, array1: T, array2: Array | float) -> T:
        array1.value.mul_(_unwrap(array2))
        return array1

    def div(self, array1: Array | float, array2: Array | float) -> Array:
        return Array(torch.div(_unwrap(array1), _unwrap(array2)))

    def idiv[T: Array](self, array1: T, array2: Array | float) -> T:
        array1.value.div_(_unwrap(array2))
        return array1

    def pow(self, array: Array, p: float) -> Array:
        return Array(torch.pow(array.value, p))

    def negative(self, array: Array) -> Array:
        return Array(torch.neg(array.value))

    def absolute(self, array: Array) -> Array:
        return Array(torch.abs(array.value))

    def sqrt(self, array: Array) -> Array:
        return Array(torch.sqrt(array.value))

    # Operators

    def sign(self, array: Array) -> Array:
        return Array(torch.sign(array.value))

    def maximum(self, array1: Array | float, array2: Array | float) -> Array:
        return Array(torch.maximum(_unwrap(array1), _unwrap(array2)))

    def argmax(self, array: Array, dim: int | None = None, keepdims: bool = False) -> Array:
        return Array(torch.argmax(array.value, dim=dim, keepdim=keepdims))

    def argmin(self, array: Array, dim: int | None = None, keepdims: bool = False) -> Array:
        return Array(torch.argmin(array.value, dim=dim, keepdim=keepdims))

    def set_item(self, array: Array, key: ArrayKey, value: Array) -> None:
        array.value[key] = value.value

    def get_item(self, array: Array, key: ArrayKey) -> Array:
        return Array(array.value[key])

    # RNG

    def set_seed(self, seed: int) -> None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        self._generator.manual_seed(seed)

    def get_rng_state(self) -> dict[str, Any]:
        state: dict[str, Any] = {
            "torch_cpu_state": torch.random.get_rng_state(),
            "torch_generator_state": self._generator.get_state(),
        }
        if torch.cuda.is_available():
            state["torch_cuda_states"] = torch.cuda.get_rng_state_all()
        return state

    def set_rng_state(self, state: dict[str, Any]) -> None:
        if "torch_cpu_state" in state:
            torch.random.set_rng_state(state["torch_cpu_state"])
        if "torch_cuda_states" in state and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(state["torch_cuda_states"])
        if "torch_generator_state" in state:
            self._generator.set_state(state["torch_generator_state"])

    def normal(self, mean: float = 0.0, std: float = 1.0, shape: tuple[int, ...] = ()) -> Array:
        return Array(
            torch.normal(mean=mean, std=std, size=shape, device=self._native_device, generator=self._generator)
        )

    def uniform(self, low: float = 0.0, high: float = 1.0, shape: tuple[int, ...] = ()) -> Array:
        rand = torch.rand(size=shape, device=self._native_device, generator=self._generator)
        return Array((high - low) * rand + low)

    def normal_like(self, array: Array, mean: float = 0.0, std: float = 1.0) -> Array:
        v = array.value
        return Array(
            torch.normal(
                mean=mean,
                std=std,
                size=tuple(v.shape),
                dtype=v.dtype,
                device=v.device,
                generator=self._generator,
            )
        )

    def uniform_like(self, array: Array, low: float = 0.0, high: float = 1.0) -> Array:
        v = array.value
        rand = torch.rand(size=tuple(v.shape), dtype=v.dtype, device=v.device, generator=self._generator)
        return Array((high - low) * rand + low)

    def choice(self, array: Array, size: int, replace: bool = True) -> Array:
        v = array.value
        weights = torch.ones(v.shape[0], device=v.device)
        indices = weights.multinomial(num_samples=size, replacement=replace, generator=self._generator)
        return Array(v[indices])


register_backend(SupportedFrameworks.PYTORCH, PyTorchBackend)
