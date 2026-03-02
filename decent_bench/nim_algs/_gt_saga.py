from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

import decent_bench.utils.algorithm_helpers as alg_helpers
import decent_bench.utils.interoperability as iop
from decent_bench.agents import Agent
from decent_bench.costs import EmpiricalRiskCost
from decent_bench.distributed_algorithms import Algorithm
from decent_bench.networks import P2PNetwork

if TYPE_CHECKING:
    from decent_bench.utils.array import Array


@dataclass(eq=False)
class GT_SAGA(Algorithm):  # noqa: N801
    """
    Gradient Tracking with SAGA variance reduction :footcite:p:`GT_SAGA_Xin_2020` :footcite:p:`GT_SAGA_Xin_2022`.

    Warning:
        GT-SAGA is only compatible with EmpiricalRiskCost. Using it with other cost types may lead to errors or
        undefined behavior.

    Args:
        iterations: Total number of iterations
        step_size: Step size for local updates, can be a constant or a function of iteration
        x0: Initial parameters (optional)
        name: Algorithm name (default "GT-SAGA")

    """

    iterations: int = 100
    step_size: float | Callable[[int], float] = 0.01
    x0: "Array | None" = None  # Initial parameters (optional)
    name: str = "GT-SAGA"

    def __post_init__(self) -> None:
        """
        Validate parameters.

        Raises:
            ValueError: If any of the parameters are invalid (e.g., non-positive iterations, local_steps,
            step_size, penalty, or alpha).

        """
        if self.iterations <= 0:
            raise ValueError("iterations must be positive")
        if isinstance(self.step_size, float) and self.step_size <= 0:
            raise ValueError("step_size must be positive")
        if callable(self.step_size):
            test_step_size = [self.step_size(k) for k in range(self.iterations)]
            if any(s <= 0 for s in test_step_size):
                raise ValueError("step_size function must return positive values for all iterations")

    def initialize(self, network: P2PNetwork) -> None:
        self.x0 = alg_helpers.zero_initialization(self.x0, network)
        self.W = network.weights

        for agent in network.agents():
            # Check that cost function supports SAGA
            if not isinstance(agent.cost, EmpiricalRiskCost):
                raise TypeError("GT-SAGA only supports EmpiricalRiskCost instances.")

            # Initialize gradient table: z_{i,j}^0 = x_i^0 for all j
            z_grads = agent.cost.gradient(self.x0, indices="all", reduction=None)

            # Initialize y_i^0 = 0_p and g_i^{-1} = 0_p
            y0 = iop.zeros_like(self.x0)
            g_minus1 = iop.zeros_like(self.x0)

            # Initialize auxiliary variables
            aux_vars = {
                "z_grads": z_grads,  # Gradient table z_{i,j}
                "y": y0,  # Gradient tracking variable y_i^0 = 0
                "g_old": g_minus1,  # Previous gradient estimator g_i^{-1} = 0
                "g": g_minus1,
            }

            # Estimate received messages using agent's own initial values
            # Since y0 = 0, we just send x_0
            y_plus_delta_g = y0

            agent.initialize(
                x=self.x0,
                aux_vars=aux_vars,
                received_msgs=dict.fromkeys(network.neighbors(agent), y_plus_delta_g),
            )

    def step(self, network: P2PNetwork, iteration: int) -> None:
        step_size = self.step_size(iteration) if callable(self.step_size) else self.step_size

        # Step 1: Select random sample and update local stochastic gradient estimator
        for i in network.active_agents(iteration):
            self._update_gradient_estimator(i)

        # Step 2: Update gradient tracker
        # y_i^{k+1} = sum_{r=1}^n w_ir (y_r^k + g_r^k - g_r^{k-1})
        for i in network.active_agents(iteration):
            # Broadcast y_i + g_i - g_i^{-1}
            y_plus_delta_g = i.aux_vars["y"] + i.aux_vars["g"] - i.aux_vars["g_old"]
            network.broadcast(i, y_plus_delta_g)

        for i in network.active_agents(iteration):
            network.receive_all(i)

        for i in network.active_agents(iteration):
            self._update_gradient_tracker(i)

        # Step 3: Update local estimate of the solution
        # x_i^{k+1} = sum_{r=1}^n w_ir (x_r^k - alpha*y_r^{k+1})
        for i in network.active_agents(iteration):
            # Broadcast x_i - alpha*y_i to reduce communication
            x_minus_alpha_y = i.x - step_size * i.aux_vars["y"]
            network.broadcast(i, x_minus_alpha_y)

        for i in network.active_agents(iteration):
            network.receive_all(i)

        for i in network.active_agents(iteration):
            self._consensus_update(i)

        # Step 4: Update gradient table for a select samples
        for i in network.active_agents(iteration):
            self._update_gradient_table(i)

    def _update_gradient_estimator(self, agent: Agent) -> None:
        """
        Update local stochastic gradient estimator using SAGA variance reduction.

        Raises:
            TypeError: If the agent's cost function is not an instance of EmpiricalRiskCost.

        """
        if not isinstance(agent.cost, EmpiricalRiskCost):
            raise TypeError("GT-SAGA is only compatible with EmpiricalRiskCost.")

        # Store old g_i for gradient tracking update
        agent.aux_vars["g_old"] = iop.copy(agent.aux_vars["g"])

        # Compute grad f_{i,tau_i}(x_i^k), gradient at current point for selected sample
        grad_current = agent.cost.gradient(agent.x)
        batch_used = agent.cost.batch_used

        # Get grad f_{i,tau_i}(z_{i,tau_i}^k), gradient at stored point for selected sample
        z_grads = iop.mean(agent.aux_vars["z_grads"][batch_used], dim=0)

        # Compute (1/m) sum_{j=1}^m grad f_{i,j}(z_{i,j}^k), average of all gradients in table
        avg_table_grad = iop.mean(agent.aux_vars["z_grads"], dim=0)

        # Update SAGA gradient estimator
        # g_i^k = grad f_{i,tau_i}(x_i) - grad f_{i,tau_i}(z_{i,tau_i}) + (1/m) sum_{j=1}^m grad f_{i,j}(z_{i,j})
        agent.aux_vars["g"] = grad_current - z_grads + avg_table_grad

    def _update_gradient_tracker(self, agent: Agent) -> None:
        """Update local gradient tracker."""
        weighted_sum = iop.sum(
            iop.stack([self.W[agent, j] * y_plus_delta_g for j, y_plus_delta_g in agent.messages.items()]),
            dim=0,
        )
        weighted_sum += self.W[agent, agent] * (agent.aux_vars["y"] + agent.aux_vars["g"] - agent.aux_vars["g_old"])
        agent.aux_vars["y"] = weighted_sum

    def _consensus_update(self, agent: Agent) -> None:
        """Update local estimate via consensus."""
        weighted_sum = iop.sum(
            iop.stack([self.W[agent, j] * x_minus_alpha_y for j, x_minus_alpha_y in agent.messages.items()]),
            dim=0,
        )
        weighted_sum += self.W[agent, agent] * (agent.x - self.step_size * agent.aux_vars["y"])
        agent.x = weighted_sum

    def _update_gradient_table(self, agent: Agent) -> None:
        """
        Update gradient table for the selected sample.

        Raises:
            TypeError: If the agent's cost function is not an instance of EmpiricalRiskCost.

        """
        if not isinstance(agent.cost, EmpiricalRiskCost):
            raise TypeError("GT-SAGA is only compatible with EmpiricalRiskCost.")

        z_grads = agent.cost.gradient(agent.x, indices="batch", reduction=None)
        batch_used = agent.cost.batch_used

        # Update the gradient table entry for the selected sample
        agent.aux_vars["z_grads"][batch_used] = z_grads
        # All other entries remain unchanged (implicit)
