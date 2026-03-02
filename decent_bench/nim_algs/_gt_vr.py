import random
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
class GT_VR(Algorithm):  # noqa: N801
    """
    GT-VR: Gradient Tracking with Variance Reduction algorithm :footcite:p:`GT_VR_Jiang_2023`.

    Warning:
        GT-VR is only compatible with EmpiricalRiskCost. Using it with other cost types may lead to errors or
        undefined behavior.

    Args:
        iterations: Total number of iterations
        step_size: Step size for primal updates, can be a constant or a function of iteration
        snapshot_prob: Probability of performing a snapshot update (P in the paper)
        x0: Initial parameters (optional)
        name: Algorithm name (default "GT-VR")

    """

    iterations: int = 100
    step_size: float | Callable[[int], float] = 0.01
    snapshot_prob: float = 0.3  # P in the algorithm
    x0: "Array | None" = None  # Initial parameters (optional)
    name: str = "GT-VR"

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
        if not 0 < self.snapshot_prob <= 1:
            raise ValueError("snapshot_prob must be in (0, 1]")

    def initialize(self, network: P2PNetwork) -> None:
        """
        Initialize agents.

        Algorithm 1, line 1

        Raises:
            TypeError: If any agent's cost function is not an instance of EmpiricalRiskCost, since GT-VR relies on
                variance reduction techniques that require access to individual sample gradients. Using GT-VR with
                incompatible cost functions may lead to errors or undefined behavior.

        """
        self.x0 = alg_helpers.zero_initialization(self.x0, network)
        self.W = network.weights

        # Update with global RNG
        random.seed(0)

        for agent in network.agents():
            # Check that cost function supports variance reduction
            if not isinstance(agent.cost, EmpiricalRiskCost):
                raise TypeError("GT-VR only supports EmpiricalRiskCost instances.")

            # Compute full gradient at initialization: grad f_i(x_i^1)
            full_grad = agent.cost.gradient(self.x0, indices="all")

            # Initialize auxiliary variables according to line 1
            aux_vars = {
                "tau": (self.x0),  # tau_i^1 = x_i^1 (for snapshot updates)
                "full_grad_tau": full_grad,  # grad f_i(tau_i) - cached to avoid recomputation
                "y": full_grad,  # y_i^1 = grad f_i(x_i^1)
                "v": full_grad,  # v_i^1 = grad f_i(x_i^1)
                "v_old": full_grad,  # Store v_i^k for gradient tracking update
            }
            agent.initialize(
                x=self.x0,
                aux_vars=aux_vars,
                received_msgs=dict.fromkeys(network.neighbors(agent), full_grad),
            )

    def step(self, network: P2PNetwork, iteration: int) -> None:
        # Main algorithm loop (line 2)
        step_size = self.step_size(iteration) if callable(self.step_size) else self.step_size

        for i in network.active_agents(iteration):
            x_minus_eta_y = i.x - step_size * i.aux_vars["y"]
            network.broadcast(i, x_minus_eta_y)

        for i in network.active_agents(iteration):
            network.receive_all(i)

        # Step 1: Update local estimate of the solution (line 3)
        for i in network.active_agents(iteration):
            self._consensus_update(i, step_size)

        # Step 2: Probabilistic snapshot update (line 4)
        # Select l_i^{k+1} ~ Bernoulli(P)
        snapshot_decisions = {}
        for i in network.active_agents(iteration):
            # Replace with global RNG
            l_i = random.random() < self.snapshot_prob
            snapshot_decisions[i] = l_i

            if l_i:  # l_i^{k+1} = 1
                # Update: tau_i^{k+1} = x_i^{k+1} and recompute full gradient
                self._snapshot_update(i)

        # Step 3: Select batch and update local gradient estimator (lines 5-6)
        for i in network.active_agents(iteration):
            self._update_gradient_estimator(i)

        # We broadcast y_i + v_i - v_old to reduce communication
        for i in network.active_agents(iteration):
            y_plus_delta_v = i.aux_vars["y"] + i.aux_vars["v"] - i.aux_vars["v_old"]
            network.broadcast(i, y_plus_delta_v)

        for i in network.active_agents(iteration):
            network.receive_all(i)

        # Step 4: Update gradient tracker (line 7)
        for i in network.active_agents(iteration):
            self._update_gradient_tracker(i)

    def _consensus_update(self, agent: Agent, step_size: float) -> None:
        """
        Update local estimate via consensus.

        Algorithm 1, line 3.

        """
        weighted_sum = iop.sum(
            iop.stack([self.W[agent, j] * x_minus_eta_y for j, x_minus_eta_y in agent.messages.items()]),
            dim=0,
        )
        weighted_sum += self.W[agent, agent] * (agent.x - step_size * agent.aux_vars["y"])
        agent.x = weighted_sum

    def _snapshot_update(self, agent: Agent) -> None:
        """
        Update snapshot point when l_i^{k+1} = 1.

        Algorithm 1, line 4.

        """
        agent.aux_vars["tau"] = iop.copy(agent.x)

        # Compute and cache the full gradient at the new snapshot point
        full_grad_tau = agent.cost.gradient(agent.aux_vars["tau"], indices="all")
        agent.aux_vars["full_grad_tau"] = full_grad_tau

    def _update_gradient_estimator(self, agent: Agent) -> None:
        """
        Update local stochastic gradient estimator with variance reduction.

        Algorithm 1, lines 5-6:

        This implements the variance reduction technique (Equation 3)

        Raises:
            TypeError: If the agent's cost is not an instance of EmpiricalRiskCost.

        """
        if not isinstance(agent.cost, EmpiricalRiskCost):
            raise TypeError("GT-VR is only compatible with EmpiricalRiskCost.")

        # Store old v_i for gradient tracking update
        agent.aux_vars["v_old"] = iop.copy(agent.aux_vars["v"])

        # Select s_i^{k+1} uniformly at random (this is done by the cost function)
        # Compute stochastic gradient at current point: grad f_{i,s_i}(x_i^{k+1})
        grad_current = agent.cost.gradient(agent.x)
        batch_indices = agent.cost.batch_used

        # Compute stochastic gradient at snapshot point: grad f_{i,s_i}(tau_i^{k+1})
        grad_snapshot = agent.cost.gradient(agent.aux_vars["tau"], indices=batch_indices)

        # Use cached full gradient at snapshot point: grad f_i(tau_i^{k+1})
        full_grad_snapshot = agent.aux_vars["full_grad_tau"]

        # Update variance-reduced gradient estimator (Equation 3)
        # v_i^{k+1} = grad f_{i,s_i}(x_i) - grad f_{i,s_i}(tau_i) + grad f_i(tau_i)
        agent.aux_vars["v"] = grad_current - grad_snapshot + full_grad_snapshot

    def _update_gradient_tracker(self, agent: Agent) -> None:
        """
        Update local gradient tracker.

        Algorithm 1, line 7:

        Note:
            We receive y_r + v_r - v_r_old directly to reduce communication.

        """
        weighted_sum = iop.sum(
            iop.stack([self.W[agent, j] * y_plus_delta_v for j, y_plus_delta_v in agent.messages.items()]),
            dim=0,
        )
        weighted_sum += self.W[agent, agent] * (agent.aux_vars["y"] + agent.aux_vars["v"] - agent.aux_vars["v_old"])
        agent.aux_vars["y"] = weighted_sum
