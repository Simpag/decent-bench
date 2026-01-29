from __future__ import annotations

from collections.abc import Iterator
from functools import cached_property
from typing import TYPE_CHECKING, Any, cast, override

from decent_bench.costs._base._sum_cost import SumCost
import decent_bench.utils.interoperability as iop
from decent_bench.costs._base._cost import Cost
from decent_bench.costs._empirical_risk._empirical_risk_cost import EmpiricalRiskCost
from decent_bench.datasets import DatasetPartition
from decent_bench.utils.logger import LOGGER
from decent_bench.utils.types import EmpiricalRiskBatchSize, EmpiricalRiskIndices, SupportedDevices, SupportedFrameworks

if TYPE_CHECKING:
    import torch

try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class IndexDataset:
    """A simple dataset wrapper to handle indexing when using a PyTorch dataloader."""

    def __init__(self, dataset: DatasetPartition):
        self.dataset = dataset

        if not hasattr(self.dataset, "__len__"):
            raise ValueError("Dataset must implement __len__ method.")

    def __iter__(self) -> Iterator[tuple[torch.Tensor, torch.Tensor, int]]:
        return iter(self.dataset)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, int]:
        return *self.dataset[idx], idx


class PyTorchCost(EmpiricalRiskCost):
    """
    A cost function wrapper for PyTorch neural networks that integrates with the distributed optimization framework.

    Supports batch-based training and gradient computation for distributed learning scenarios.
    """

    def __init__(
        self,
        dataset: DatasetPartition,
        model: torch.nn.Module,
        loss_fn: torch.nn.Module,
        final_activation: torch.nn.Module | None = None,
        *,
        batch_size: EmpiricalRiskBatchSize = "all",
        device: SupportedDevices = SupportedDevices.CPU,
        use_dataloader: bool = False,
        dataloader_kwargs: dict | None = None,
        load_dataset: bool = True,
        compile_model: bool = False,
        compile_kwargs: dict | None = None,
    ):
        """
        Initialize the neural network cost function.

        Args:
            dataset (DatasetPartition):
                Dataset partition containing features and targets.
                Transformations should be applied beforehand such as converting to tensors.
                See torch.utils.data.Dataset for details.
            model (torch.nn.Module): PyTorch neural network model.
            loss_fn: (torch.nn.Module): PyTorch loss function.
            final_activation (torch.nn.Module | None): Optional final activation layer to apply after
                model output when predicting targets. E.g., argmax if classification and model outputs logits.
            batch_size (EmpiricalRiskBatchSize): Size of mini-batches for stochastic methods, or "all" for full-batch.
            device (SupportedDevices): Device to run computations on.
            use_dataloader (bool): Whether to use DataLoader for batching.
                Can be beneficial for large datasets to avoid loading all data into memory.
            dataloader_kwargs (dict | None): Additional arguments for the DataLoader.
            load_dataset (bool): If True, load the entire dataset into memory to optimize data access.
                This may lead to major speedups if the dataset is lazily loaded.
                May increase memory usage if the dataset is lazily loaded, set to False if memory is an issue.
            compile_model (bool): Whether to compile the model using torch.compile for performance.
                May improve speed after warm-up. Might need to try different modes based on the model and OS,
                use compile_kwargs. See https://pytorch.org/docs/stable/generated/torch.compile.html for details.

        Raises:
            ImportError: If PyTorch is not available
            ValueError: If batch_size is larger than the number of samples in the dataset

        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is not available. Please install PyTorch to use PyTorchCost.")

        if isinstance(batch_size, int) and (batch_size <= 0 or batch_size > len(dataset)):
            raise ValueError(
                f"Batch size must be positive and at most the number of samples, "
                f"got: {batch_size} and number of samples is: {len(dataset)}."
            )
        if isinstance(batch_size, str) and batch_size != "all":
            raise ValueError(f"Invalid batch size string. Supported value is 'all', got {batch_size}.")

        self.model = model
        self.loss_fn = loss_fn
        self.final_activation = final_activation if final_activation is not None else torch.nn.Identity()
        self._batch_size = self.n_samples if batch_size == "all" else batch_size
        self._device = device
        self._use_dataloader = use_dataloader
        self._dataloader_kwargs = dataloader_kwargs if dataloader_kwargs is not None else {}
        self._load_dataset = load_dataset
        self._compile_model = compile_model
        self._compile_kwargs = compile_kwargs if compile_kwargs is not None else {}

        if self._load_dataset:
            # Loads the dataset into memory in case it is lazily loaded
            self.dataset = IndexDataset([(x, y) for x, y in dataset])
        else:
            self.dataset = IndexDataset(dataset)

        self._pytorch_device: str = iop.device_to_framework_device(device, framework=self.framework)
        self.model = self.model.to(self._pytorch_device)
        self.loss_fn = self.loss_fn.to(self._pytorch_device)

        if self._compile_model:
            torch.set_float32_matmul_precision("high")  # For better torch.compile performance
            self.model = cast("torch.nn.Module", torch.compile(self.model, **self._compile_kwargs))

        self._dataloader: torch.utils.data.DataLoader | None = None
        self._last_batch_used = []  # Pre-allocate list for last used batch for efficiency in _get_batch_data

        # Store parameter shapes for flattening/unflattening
        self.param_shapes = [p.shape for p in self.model.parameters()]
        self.param_sizes = [p.numel() for p in self.model.parameters()]
        self.total_params = sum(self.param_sizes)
        self.param_names = [n for n, _ in self.model.named_parameters()]
        self.param_offsets = torch.cumsum(torch.tensor([0, *self.param_sizes[:-1]]), dim=0).tolist()

    @property
    def shape(self) -> tuple[int, ...]:
        return (self.total_params,)

    @property
    def framework(self) -> SupportedFrameworks:
        return SupportedFrameworks.TORCH

    @property
    def device(self) -> SupportedDevices:
        return self._device

    @property
    def n_samples(self) -> int:
        return len(self.dataset)

    @property
    def batch_size(self) -> int:
        """Size of batches used for stochastic training."""
        return self._batch_size

    @property
    def m_smooth(self) -> float:
        return float("nan")

    @property
    def m_cvx(self) -> float:
        return float("nan")

    @cached_property
    @override
    def _rand(self) -> torch.Generator:  # type: ignore[override]
        return torch.Generator(device="cpu").manual_seed(0)  # Later replace with global rng

    def _clean(self) -> None:
        """Clean up cache."""
        self._last_batch_x = torch.empty(0)
        self._last_batch_y = torch.empty(0)

    def _set_model_parameters(self, x: torch.Tensor) -> None:
        """
        Set model parameters from a tensor.

        Args:
            x (torch.Tensor): Flattened parameter tensor.

        Raises:
            ValueError: If the size of x does not match the total number of model parameters.

        """
        if x.numel() != self.total_params:
            raise ValueError(
                f"Parameter vector size {x.numel()} does not match total model parameters {self.total_params}"
            )

        # Unflatten the parameter vector and set model parameters
        start_idx = 0
        with torch.no_grad():
            for param, size, shape in zip(self.model.parameters(), self.param_sizes, self.param_shapes, strict=True):
                end_idx = start_idx + size
                param.data = x[start_idx:end_idx].reshape(shape).to(param.device)
                start_idx = end_idx

    def _get_model_parameters(self) -> torch.Tensor:
        """Get model parameters as a flattened tensor."""
        params = [p.detach().flatten() for p in self.model.parameters()]
        return cast("torch.Tensor", torch.cat(params).to(self._pytorch_device))

    def _init_dataloader(self) -> None:
        def _collate_xy_idx(
            batch: list[tuple[torch.Tensor, torch.Tensor, int]],
        ) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
            xs, ys, idx = zip(*batch, strict=True)
            return torch.stack(xs), torch.stack(ys), list(idx)

        self._dataloader_kwargs.setdefault("shuffle", True)
        self._dataloader = torch.utils.data.DataLoader(
            cast("torch.utils.data.Dataset[Any]", self.dataset),
            batch_size=self.batch_size,
            generator=self._rand,
            collate_fn=_collate_xy_idx,
            **self._dataloader_kwargs,
        )
        self._dataloader_iter = iter(self._dataloader)

    @iop.autodecorate_cost_method(EmpiricalRiskCost.predict)
    def predict(self, x: torch.Tensor, data: list[torch.Tensor]) -> list[torch.Tensor]:
        """
        Make predictions at x on the given data.

        Args:
            x: Point to make predictions at.
            data: List of torch.Tensor containing features to make predictions on.

        Returns:
            Predicted targets as an array

        """
        self._set_model_parameters(x)
        self.model.eval()
        with torch.no_grad():
            inputs = torch.stack(data).to(self._pytorch_device)
            outputs: torch.Tensor = self.model(inputs)
            outputs = self.final_activation(outputs)

        return outputs.detach().cpu().tolist()

    @iop.autodecorate_cost_method(EmpiricalRiskCost.function)
    def function(self, x: torch.Tensor, indices: EmpiricalRiskIndices = "batch") -> float:
        self._set_model_parameters(x)
        self.model.eval()

        batch_x, batch_y = self._get_batch_data(indices)

        with torch.no_grad():
            outputs = self.model(batch_x)
            loss: torch.Tensor = self.loss_fn(outputs, batch_y)

        return float(loss.cpu().item())

    @iop.autodecorate_cost_method(EmpiricalRiskCost.gradient)
    def gradient(
        self,
        x: torch.Tensor,
        indices: EmpiricalRiskIndices = "batch",
    ) -> torch.Tensor | dict[int, torch.Tensor]:
        self._set_model_parameters(x)
        self.model.train()

        batch_x, batch_y = self._get_batch_data(indices)

        # Forward pass
        outputs = self.model(batch_x)
        loss = self.loss_fn(outputs, batch_y)

        # Compute gradients using torch.autograd.grad (doesn't modify model parameters)
        model_params = self.model.parameters()
        gradients = torch.autograd.grad(
            loss,
            model_params,
            create_graph=False,
            retain_graph=False,
            allow_unused=True,
        )

        grads = [
            g.reshape(-1) if g is not None else torch.zeros_like(p)
            for p, g in zip(self.model.parameters(), gradients, strict=True)
        ]

        # Return concatenated gradient tensor
        return torch.cat(grads)

    @iop.autodecorate_cost_method(EmpiricalRiskCost.per_sample_gradients)
    def per_sample_gradients(self, x: torch.Tensor, indices: EmpiricalRiskIndices = "batch") -> dict[int, torch.Tensor]:
        """Compute per-sample gradients for the specified indices. May need to batch calls due to memory constraints."""
        # Credit: https://docs.pytorch.org/tutorials/intermediate/per_sample_grads.html
        self._init_per_sample_grad()
        self._set_model_parameters(x)
        self.model.train()

        batch_x, batch_y = self._get_batch_data(indices)

        params = {k: v.detach() for k, v in self.model.named_parameters()}
        buffers = {k: v.detach() for k, v in self.model.named_buffers()}

        ft_per_sample_grads = self._ft_compute_sample_grad(params, buffers, batch_x, batch_y)

        # Collect gradients and flatten them into a single tensor
        x_shape = batch_x.shape[0]
        dtype = next(self.model.parameters()).dtype
        with torch.no_grad():
            flat_grads = torch.empty((x_shape, self.total_params), device=self._pytorch_device, dtype=dtype)
            for name, off, size in zip(self.param_names, self.param_offsets, self.param_sizes, strict=True):
                g = ft_per_sample_grads[name].reshape(x_shape, size)
                flat_grads[:, off : off + size] = g

        return {idx: flat_grads[i] for i, idx in enumerate(self.batch_used)}

    @iop.autodecorate_cost_method(EmpiricalRiskCost.hessian)
    def hessian(self, x: torch.Tensor, indices: EmpiricalRiskIndices = "batch") -> torch.Tensor:
        """
        Compute the Hessian matrix.

        Note:
            This is computationally expensive for neural networks and typically not used.

        Raises:
            NotImplementedError: Always raised to indicate Hessian computation is not implemented.

        """
        raise NotImplementedError("Hessian computation is not implemented for PyTorchCost.")

    @iop.autodecorate_cost_method(EmpiricalRiskCost.proximal)
    def proximal(self, x: torch.Tensor, rho: float) -> torch.Tensor:
        """
        Compute the proximal operator.

        Note:
            This is computationally expensive for neural networks and typically not used.

        Raises:
            NotImplementedError: Always raised to indicate proximal computation is not implemented.

        """
        raise NotImplementedError("Proximal operator is not implemented for NeuralNetworkCostFunction.")

    @override
    def _sample_batch_indices(self, indices: EmpiricalRiskIndices = "batch") -> list[int]:
        raise NotImplementedError("_sample_batch_indices is not used in PyTorchCost, implemented in _get_batch_data.")

    def _get_batch_data(self, indices: EmpiricalRiskIndices = "batch") -> tuple[torch.Tensor, torch.Tensor]:
        """Get data for a batch. Returns features and targets tensors."""
        batch_x: torch.Tensor | None = None
        batch_y: torch.Tensor | None = None
        batch_idx: list[int] | None = None
        if isinstance(indices, str):
            if indices == "batch":
                if self.batch_size < self.n_samples:
                    if self._use_dataloader:
                        if self._dataloader is None:
                            self._init_dataloader()

                        try:
                            batch_x, batch_y, batch_idx = next(self._dataloader_iter)
                        except StopIteration:
                            # Restart the iterator if we reach the end
                            self._dataloader_iter = iter(self._dataloader)
                            batch_x, batch_y, batch_idx = next(self._dataloader_iter)
                    else:
                        indices = torch.randperm(self.n_samples, generator=self._rand)[: self.batch_size].tolist()
                else:
                    # Use full dataset
                    indices = list(range(self.n_samples))
            elif indices == "all":
                indices = list(range(self.n_samples))
            else:
                raise ValueError(f"Invalid indices string: {indices}. Only 'all' and 'batch' are supported.")

        if isinstance(indices, int):
            indices = [indices]

        if isinstance(indices, list):
            if len(indices) == len(self.batch_used) and indices == self.batch_used:
                # Use cached batch so we don't have to re-stack
                return self._last_batch_x, self._last_batch_y

            batch = (
                self.dataset
                if len(indices) == len(self.dataset) and self._load_dataset
                else [self.dataset[i] for i in indices]
            )
            batch_x, batch_y, batch_idx = zip(*batch, strict=True)
            batch_x = torch.stack(batch_x)
            batch_y = torch.stack(batch_y)

        if batch_x is None or batch_y is None or batch_idx is None:
            raise RuntimeError("Batch data could not be retrieved. Please report this error.")

        self._last_batch_used = list(batch_idx)
        self._last_batch_x = batch_x.to(self._pytorch_device, non_blocking=True)
        self._last_batch_y = batch_y.to(self._pytorch_device, non_blocking=True)

        return self._last_batch_x, self._last_batch_y

    def _init_per_sample_grad(self) -> None:
        """Initialize per-sample gradient function using functorch."""
        if hasattr(self, "_ft_compute_sample_grad"):
            return  # Already initialized

        def compute_loss(
            params: dict[str, torch.Tensor],
            buffers: dict[str, torch.Tensor],
            sample: torch.Tensor,
            target: torch.Tensor,
        ) -> torch.Tensor:
            batch = sample.unsqueeze(0)
            targets = target.unsqueeze(0)

            predictions = torch.func.functional_call(self.model, (params, buffers), (batch,))
            loss: torch.Tensor = self.loss_fn(predictions, targets)
            return loss

        self._ft_compute_grad = torch.func.grad(compute_loss)
        self._ft_compute_sample_grad = torch.func.vmap(self._ft_compute_grad, in_dims=(None, None, 0, 0))

        if not self._compile_model:
            return

        try:
            self._ft_compute_sample_grad = torch.compile(self._ft_compute_sample_grad, **self._compile_kwargs)
        except Exception as e:
            LOGGER.warning(f"Error compiling per-sample gradient function: {e}\n\nContinuing without compilation.")

    def __add__(self, other: Cost) -> Cost:
        if self.shape != other.shape:
            raise ValueError(f"Mismatching domain shapes: {self.shape} vs {other.shape}")
        return SumCost([self, other])


class SimpleMLP(torch.nn.Module):
    """A simple multi-layer perceptron for demonstration."""

    def __init__(
        self,
        input_size: int,
        hidden_sizes: list[int],
        output_size: int,
        activation: str = "relu",
        output_activation: str | None = None,
    ):
        super().__init__()

        layers = []
        prev_size = input_size

        # Hidden layers
        for hidden_size in hidden_sizes:
            layers.append(torch.nn.Linear(prev_size, hidden_size))
            if activation == "relu":
                layers.append(torch.nn.ReLU())
            elif activation == "tanh":
                layers.append(torch.nn.Tanh())
            elif activation == "sigmoid":
                layers.append(torch.nn.Sigmoid())
            prev_size = hidden_size

        # Output layer
        layers.append(torch.nn.Linear(prev_size, output_size))
        if output_activation is not None:
            if output_activation == "relu":
                layers.append(torch.nn.ReLU())
            elif output_activation == "tanh":
                layers.append(torch.nn.Tanh())
            elif output_activation == "sigmoid":
                layers.append(torch.nn.Sigmoid())

        self.network = torch.nn.Sequential(*layers)

    def forward(self, x) -> torch.Tensor:
        return self.network(x)


def create_simple_mlp(
    input_size: int,
    hidden_sizes: list[int],
    output_size: int,
    activation: str = "relu",
    output_activation: str | None = None,
) -> SimpleMLP:
    """
    Factory function to create a simple MLP.

    Compilation can be toggled to improve performance, does require a certain "warm-up" time.
    See pytorch documentation for details and OS restrictions.
    """  # noqa: D401
    return SimpleMLP(input_size, hidden_sizes, output_size, activation, output_activation)


if __name__ == "__main__":
    # Simple test case
    from rich.progress import track
    import time
    from torch import nn
    import numpy as np

    mnist_train = [
        (
            torch.tensor(x, dtype=torch.float32),
            torch.tensor(y, dtype=torch.long),
        )
        for x, y in ((np.random.rand(3), np.random.randint(0, 2)) for _ in range(256))
    ]

    # Create a simple MLP model
    model = create_simple_mlp(input_size=3, hidden_sizes=[64, 128, 128, 64, 32], output_size=2)

    # Create the cost function wrapper
    cost_function = PyTorchCost(
        model=model,
        dataset=mnist_train,
        loss_fn=nn.CrossEntropyLoss(),
        batch_size=32,  # 32,
        device=SupportedDevices.GPU,
    )

    # Test batch getter
    x = torch.randn(cost_function.total_params, dtype=torch.float32, device=cost_function._pytorch_device)
    grad1 = cost_function.gradient(x)
    print(f"Gradient shape: {grad1.shape}")
    print(f"Gradient norm: {grad1.norm().item():.4f}")
    batch = cost_function.batch_used

    print(f"Get parameters works: {x.equal(cost_function._get_model_parameters())}")

    # Test evaluation and gradient computation
    grad2 = cost_function.gradient(x, batch)
    grad3 = cost_function.gradient(x, batch)

    print(f"Gradient equal: {grad1.equal(grad2)}, {grad2.equal(grad3)}")
    print(f"Gradient device: {grad1.device}, {grad2.device}, {grad3.device}")
    print(f"X device: {x.device}")
    print(f"Shape of gradient: {grad1.shape}")

    # # Test speed
    # import time

    # start_time = time.time()
    # for _ in range(500):
    #     cost_function.gradient(x)
    # end_time = time.time()
    # print(f"Time for 500 gradient computations: {end_time - start_time:.4f} seconds")

    # start_time = time.time()
    # for _ in range(500):
    #     cost_function.gradient(x, batch)
    # end_time = time.time()
    # print(
    #     f"Time for 500 gradient computations (with specified indicies): {end_time - start_time:.4f} seconds"
    # )

    z, y = cost_function._get_batch_data([0, 1, 2, 3])

    print(f"Batch x shape: {z.shape}, y shape: {y.shape}")

    indices = list(range(128))

    t1 = list()
    t2 = list()
    for i in track(range(100)):
        t = time.time()
        efficient_ind = cost_function.per_sample_gradients(x, indices)
        t1.append(time.time() - t)
        t = time.time()
        slow_ind = {i: cost_function.gradient(x, [i]) for i in indices}
        t2.append(time.time() - t)

    t1 = t1[25:]
    t2 = t2[25:]

    rtol = 5e-3
    atol = 1e-4
    print(f"Time for individual gradients: {sum(t1) / len(t1):.4f} seconds")
    print(f"Time for slow individual gradients: {sum(t2) / len(t2):.4f} seconds")
    print(f"Keys equal: {list(efficient_ind.keys()) == list(slow_ind.keys())}")
    print(
        f"Values are close: {all([torch.allclose(efficient_ind[i], slow_ind[i], rtol=rtol, atol=atol) for i in indices])}"
    )
    print([torch.allclose(efficient_ind[i], slow_ind[i], rtol=rtol, atol=atol) for i in indices])
    different = [torch.allclose(efficient_ind[i], slow_ind[i], rtol=rtol, atol=atol) for i in indices]
    different = [i for i, eq in enumerate(different) if not eq]
    if len(different) == 0:
        print("All gradients match!")
        exit(0)
    for i in range(len(efficient_ind[different[0]])):
        if torch.isclose(
            efficient_ind[different[0]][i],
            slow_ind[different[0]][i],
            rtol=rtol,
            atol=atol,
        ):
            continue
        print(
            f"Index {i}: efficient {efficient_ind[different[0]][i]}, slow {slow_ind[different[0]][i]}; difference {efficient_ind[different[0]][i] - slow_ind[different[0]][i]}; relative difference {abs(efficient_ind[different[0]][i] - slow_ind[different[0]][i]) / max(slow_ind[different[0]][i], efficient_ind[different[0]][i]) if max(slow_ind[different[0]][i], efficient_ind[different[0]][i]) != 0 else 1e-8}"
        )
