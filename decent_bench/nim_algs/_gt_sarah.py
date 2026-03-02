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
class GT_SARAH(Algorithm):  # noqa: N801
    """
    GT-SARAH: Gradient Tracking with SARAH variance reduction :footcite:p:`GT_SARAH_Xin_2021`.

    Warning:
        GT-SARAH is only compatible with EmpiricalRiskCost. Using it with other cost types may lead to errors or
        undefined behavior.

    Args:
        iterations: Total number of outer loops (S)
        local_steps: Number of inner loop iterations (q)
        step_size: Step size (alpha) for updates, can be a constant or a function of iteration
        x0: Initial parameters (optional)
        name: Algorithm name (default "GT-SARAH")

    """

    iterations: int = 100  # S: number of outer loops
    local_steps: int = 5  # q: number of inner loop iterations
    step_size: float | Callable[[int], float] = 0.01  # alpha: step size
    x0: "Array | None" = None  # Initial parameters (optional)
    name: str = "GT-SARAH"

    def __post_init__(self) -> None:
        """
        Validate parameters.

        Raises:
            ValueError: If any of the parameters are invalid (e.g., non-positive iterations, local_steps,
            step_size, penalty, or alpha).

        """
        if self.iterations <= 0:
            raise ValueError("iterations must be positive")
        if self.local_steps <= 0:
            raise ValueError("local_steps must be positive")
        if isinstance(self.step_size, float) and self.step_size <= 0:
            raise ValueError("step_size must be positive")
        if callable(self.step_size):
            test_step_size = [self.step_size(k) for k in range(self.iterations)]
            if any(s <= 0 for s in test_step_size):
                raise ValueError("step_size function must return positive values for all iterations")

    def initialize(self, network: P2PNetwork) -> None:
        """
        Initialize agents with x_i^{0,1}, y_i^{0,1}, v_i^{-1,1}.

        Raises:
            TypeError: If any agent's cost function is not an instance of EmpiricalRiskCost.

        """
        self.x0 = alg_helpers.zero_initialization(self.x0, network)
        self.W = network.weights

        for i in network.agents():
            # Check that cost function supports variance reduction
            if not isinstance(i.cost, EmpiricalRiskCost):
                raise TypeError("GT-SARAH only supports EmpiricalRiskCost instances.")

            # Initialize y_i^{0,1} = 0 and v_i^{-1,1} = 0
            y0 = iop.zeros_like(self.x0)
            v_minus1 = iop.zeros_like(self.x0)

            # Initialize auxiliary variables
            aux_vars = {
                "y": y0,  # Gradient tracking variable y_i^{0,1} = 0
                "v": v_minus1,  # Current SARAH estimator
                "v_prev": v_minus1,  # v_i^{-1,1} = 0 (for outer loop tracking)
                "x_prev": self.x0,  # Store x_{t-1} for SARAH
            }
            # Estimate received messages using agent's own initial values
            i.initialize(
                x=self.x0,
                aux_vars=aux_vars,
                received_msgs=dict.fromkeys(network.neighbors(i), y0),
            )

    def step(self, network: P2PNetwork, iteration: int) -> None:
        step_size = self.step_size(iteration) if callable(self.step_size) else self.step_size

        # Step 1: Compute full gradient (batch gradient computation)
        for i in network.active_agents(iteration):
            self._compute_batch_grad(i)

        # Step 2: Update gradient tracker
        for i in network.active_agents(iteration):
            network.broadcast(i, i.aux_vars["y"])

        for i in network.active_agents(iteration):
            network.receive_all(i)

        for i in network.active_agents(iteration):
            self._update_gradient_tracker(i)

        # Step 3: Update state
        for i in network.active_agents(iteration):
            network.broadcast(i, i.x)

        for i in network.active_agents(iteration):
            network.receive_all(i)

        for i in network.active_agents(iteration):
            self._state_update(i, step_size)

        self._inner_loop(network, iteration, step_size)

    def _compute_batch_grad(self, agent: Agent) -> None:
        """
        Compute full gradient at the beginning of each outer loop.

        Algorithm 2.1, line 2:
        """
        agent.aux_vars["v_prev"] = iop.copy(agent.aux_vars["v"])
        grad = agent.cost.gradient(agent.x, indices="all")

        # Update v_i^{0,s} = grad f_i(x_i^{0,s})
        agent.aux_vars["v"] = grad

    def _update_sarah_estimator(self, agent: Agent) -> None:
        """
        Update SARAH variance-reduced gradient estimator.

        Algorithm 2.1, line 8.

        Raises:
            TypeError: If the agent's cost function is not an instance of EmpiricalRiskCost.

        """
        if not isinstance(agent.cost, EmpiricalRiskCost):
            raise TypeError("GT-SARAH only supports EmpiricalRiskCost instances.")

        # Store previous inner loop gradient for tracking update
        agent.aux_vars["v_prev"] = iop.copy(agent.aux_vars["v"])

        # Compute (1/B) sum_{l=1}^B grad f_{i,tau_l}(x_i^{t,s})
        grad_current = agent.cost.gradient(agent.x)
        batch_used = agent.cost.batch_used

        # Compute (1/B) sum_{l=1}^B grad f_{i,tau_l}(x_i^{t-1,s})
        grad_prev = agent.cost.gradient(agent.aux_vars["x_prev"], indices=batch_used)

        # SARAH update: v_i^{t,s} = (grad f_i(x_i, xi) - grad f_i(x_prev, xi)) + v_i^{t-1,s}
        agent.aux_vars["v"] = grad_current - grad_prev + agent.aux_vars["v_prev"]

    def _update_gradient_tracker(self, agent: Agent) -> None:
        """
        Update gradient tracker at the beginning of outer loop.

        Algorithm 2.1, line 3 and 9.

        """
        weighted_sum = iop.sum(
            iop.stack([self.W[agent, j] * y for j, y in agent.messages.items()]),
            dim=0,
        )
        weighted_sum += self.W[agent, agent] * agent.aux_vars["y"]
        agent.aux_vars["y"] = weighted_sum + agent.aux_vars["v"] - agent.aux_vars["v_prev"]

    def _state_update(self, agent: Agent, step_size: float) -> None:
        """
        Update local estimate via consensus.

        Algorithm 2.1, lines 4 and 10:

        """
        agent.aux_vars["x_prev"] = iop.copy(agent.x)
        weighted_sum = iop.sum(
            iop.stack([self.W[agent, j] * x for j, x in agent.messages.items()]),
            dim=0,
        )
        weighted_sum += self.W[agent, agent] * agent.x
        agent.x = weighted_sum - step_size * agent.aux_vars["y"]

    def _inner_loop(self, network: P2PNetwork, iteration: int, step_size: float) -> None:
        """
        Inner loop of GT-SARAH.

        Algorithm 2.1, lines 7-10.
        """
        for _ in range(self.local_steps):
            # Step 4: SARAH variance reduction
            for i in network.active_agents(iteration):
                self._update_sarah_estimator(i)  # line 8

            # Step 5: Update gradient tracker (inner loop)
            for i in network.active_agents(iteration):
                network.broadcast(i, i.aux_vars["y"])

            for i in network.active_agents(iteration):
                network.receive_all(i)

            for i in network.active_agents(iteration):
                self._update_gradient_tracker(i)  # line 9

            # Step 6: Update state (inner loop)
            for i in network.active_agents(iteration):
                network.broadcast(i, i.x)

            for i in network.active_agents(iteration):
                network.receive_all(i)

            for i in network.active_agents(iteration):
                self._state_update(i, step_size)
