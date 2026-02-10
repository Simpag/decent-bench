"""Utilities for checkpointing benchmark execution state.

This module provides the CheckpointManager class for saving and loading benchmark execution state,
enabling resumption of interrupted benchmark runs.
"""

import json
import pickle
import shutil
from pathlib import Path
from typing import Any

from decent_bench.distributed_algorithms import Algorithm
from decent_bench.networks import P2PNetwork
from decent_bench.utils.logger import LOGGER


class CheckpointManager:
    """
    Manages checkpoint directory structure and file operations for benchmark execution.

    The CheckpointManager creates and maintains a hierarchical directory structure for storing
    checkpoint data during benchmark execution. This allows benchmarks to be resumed if interrupted,
    and provides incremental saving of results as trials complete.

    Directory Structure:
        The checkpoint directory is organized as follows::

            checkpoint_dir/
            ├── metadata.json                 # Run configuration and algorithm metadata
            ├── initial_network.pkl           # Initial network state (before any trials)
            └── algorithm_0/                  # Directory for first algorithm
                ├── trial_0/                  # Directory for trial 0
                │   ├── iter_0000100.pkl     # Network state at iteration 100
                │   ├── iter_0000200.pkl     # Network state at iteration 200
                │   ├── algorithm_state.pkl  # Algorithm object state (updated each checkpoint)
                │   ├── progress.json        # {"last_completed_iteration": N}
                │   ├── final_result.pkl     # Final network state (when trial completes)
                │   └── complete.marker      # Empty marker file indicating trial completion
                ├── trial_1/
                │   └── ...
                └── trial_N/
                    └── ...

    File Descriptions:
        - **metadata.json**: Benchmark configuration (n_trials, checkpoint_step) and
          algorithm information (name, iterations, index).
        - **initial_network.pkl**: Starting network state before any algorithm execution.
        - **iter_NNNNNNN.pkl**: Network state snapshots at specific iterations during execution.
        - **algorithm_state.pkl**: Algorithm object with internal state at last checkpoint.
        - **progress.json**: Tracks the last completed iteration within a trial.
        - **final_result.pkl**: Final network state after trial completion.
        - **complete.marker**: Empty file flag indicating completed trials.

    Thread Safety:
        - Each trial writes to its own directory, avoiding write conflicts.
        - Completed trial results are loaded read-only.
        - Metadata is written once at initialization.

    Args:
        checkpoint_dir: Path to the checkpoint directory.

    Example:
        >>> manager = CheckpointManager("./my_checkpoints")
        >>> if manager.is_empty():
        ...     manager.initialize(algorithms, {"n_trials": 30})
        ...     manager.save_initial_network(initial_network)
        >>> manager.save_checkpoint(alg_idx=0, trial=0, iteration=100, algorithm=alg, network=net)
        >>> if manager.is_trial_complete(alg_idx=0, trial=0):
        ...     result = manager.load_trial_result(alg_idx=0, trial=0)

    """

    def __init__(self, checkpoint_dir: str | Path):
        """Initialize CheckpointManager with a checkpoint directory path."""
        self.checkpoint_dir = Path(checkpoint_dir)

    def is_empty(self) -> bool:
        """Check if checkpoint directory is empty or doesn't exist."""
        if not self.checkpoint_dir.exists():
            return True
        return not any(self.checkpoint_dir.iterdir())

    def initialize(self, algorithms: list[Algorithm], benchmark_metadata: dict[str, Any]) -> None:
        """
        Initialize checkpoint directory structure for a new benchmark run.

        Args:
            algorithms: List of Algorithm objects to be benchmarked.
            benchmark_metadata: Benchmark configuration (n_trials, checkpoint_step).

        """
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Save metadata
        metadata = {
            "benchmark_metadata": benchmark_metadata,
            "algorithms": [
                {
                    "name": alg.name,
                    "iterations": alg.iterations,
                    "index": idx,
                }
                for idx, alg in enumerate(algorithms)
            ],
        }
        self._save_metadata(metadata)

        # Create algorithm directories
        for idx in range(len(algorithms)):
            self._get_algorithm_dir(idx).mkdir(parents=True, exist_ok=True)

    def save_initial_network(self, network: P2PNetwork) -> None:
        """Save initial network state before any trials run."""
        initial_path = self.checkpoint_dir / "initial_network.pkl"
        with initial_path.open("wb") as f:
            pickle.dump(network, f)
        LOGGER.debug(f"Saved initial network to {initial_path}")

    def load_initial_network(self) -> P2PNetwork:
        """
        Load initial network state from checkpoint.

        Returns:
            P2PNetwork object representing the initial network state.

        Raises:
            FileNotFoundError: If initial_network.pkl doesn't exist.

        """
        initial_path = self.checkpoint_dir / "initial_network.pkl"
        with initial_path.open("rb") as f:
            return pickle.load(f)

    def save_checkpoint(
        self, alg_idx: int, trial: int, iteration: int, algorithm: Algorithm, network: P2PNetwork
    ) -> None:
        """
        Save checkpoint for a specific algorithm trial at a given iteration.

        Saves three files: network state, algorithm state, and progress tracking.

        Args:
            alg_idx: Algorithm index (0-based).
            trial: Trial number (0-based).
            iteration: Current iteration number.
            algorithm: Algorithm object with current internal state.
            network: P2PNetwork object with current agent states and metrics.

        """
        trial_dir = self._get_trial_dir(alg_idx, trial)
        trial_dir.mkdir(parents=True, exist_ok=True)

        # Save network state
        network_path = trial_dir / f"iter_{iteration:07d}.pkl"
        with network_path.open("wb") as f:
            pickle.dump(network, f)

        # Save algorithm state
        alg_path = trial_dir / "algorithm_state.pkl"
        with alg_path.open("wb") as f:
            pickle.dump(algorithm, f)

        # Update progress
        progress = {"last_completed_iteration": iteration}
        progress_path = trial_dir / "progress.json"
        with progress_path.open("w", encoding="utf-8") as f:
            json.dump(progress, f)

        LOGGER.debug(f"Saved checkpoint: alg={alg_idx}, trial={trial}, iter={iteration}")

    def load_checkpoint(self, alg_idx: int, trial: int) -> tuple[Algorithm, P2PNetwork, int] | None:
        """
        Load the latest checkpoint for a specific algorithm trial.

        Args:
            alg_idx: Algorithm index (0-based).
            trial: Trial number (0-based).

        Returns:
            Tuple of (algorithm, network, last_iteration) or None if no checkpoint exists.
            Execution should resume from iteration (last_iteration + 1).

        """
        trial_dir = self._get_trial_dir(alg_idx, trial)
        progress_path = trial_dir / "progress.json"

        if not progress_path.exists():
            return None

        # Load progress
        with progress_path.open(encoding="utf-8") as f:
            progress = json.load(f)
        last_iteration = progress["last_completed_iteration"]

        # Load algorithm state
        alg_path = trial_dir / "algorithm_state.pkl"
        with alg_path.open("rb") as f:
            algorithm = pickle.load(f)

        # Load network state
        network_path = trial_dir / f"iter_{last_iteration:07d}.pkl"
        with network_path.open("rb") as f:
            network = pickle.load(f)

        LOGGER.debug(f"Loaded checkpoint: alg={alg_idx}, trial={trial}, iter={last_iteration}")
        return algorithm, network, last_iteration

    def cleanup_old_checkpoints(self, alg_idx: int, trial: int, keep_n: int) -> None:
        """
        Remove old iteration checkpoint files, keeping only the most recent N.

        Args:
            alg_idx: Algorithm index (0-based).
            trial: Trial number (0-based).
            keep_n: Number of recent iteration checkpoints to keep.

        """
        trial_dir = self._get_trial_dir(alg_idx, trial)
        if not trial_dir.exists():
            return

        # Find all iteration checkpoint files
        checkpoint_files = sorted(trial_dir.glob("iter_*.pkl"))

        # Remove older checkpoints
        if len(checkpoint_files) > keep_n:
            for file_to_remove in checkpoint_files[:-keep_n]:
                file_to_remove.unlink()
                LOGGER.debug(f"Removed old checkpoint: {file_to_remove}")

    def mark_trial_complete(self, alg_idx: int, trial: int, final_network: P2PNetwork) -> None:
        """
        Mark a trial as complete and save final result.

        Args:
            alg_idx: Algorithm index (0-based).
            trial: Trial number (0-based).
            final_network: Final P2PNetwork state after all iterations complete.

        """
        trial_dir = self._get_trial_dir(alg_idx, trial)
        trial_dir.mkdir(parents=True, exist_ok=True)

        # Save final result
        final_path = trial_dir / "final_result.pkl"
        with final_path.open("wb") as f:
            pickle.dump(final_network, f)

        # Mark as complete
        complete_path = trial_dir / "complete.marker"
        complete_path.touch()

        LOGGER.debug(f"Marked trial complete: alg={alg_idx}, trial={trial}")

    def is_trial_complete(self, alg_idx: int, trial: int) -> bool:
        """
        Check if a trial has been completed.

        Args:
            alg_idx: Algorithm index (0-based).
            trial: Trial number (0-based).

        Returns:
            True if the trial has completed, False otherwise.

        """
        trial_dir = self._get_trial_dir(alg_idx, trial)
        return (trial_dir / "complete.marker").exists()

    def load_trial_result(self, alg_idx: int, trial: int) -> P2PNetwork:
        """
        Load final result of a completed trial.

        Args:
            alg_idx: Algorithm index (0-based).
            trial: Trial number (0-based).

        Returns:
            P2PNetwork object with final state after all iterations.

        """
        trial_dir = self._get_trial_dir(alg_idx, trial)
        final_path = trial_dir / "final_result.pkl"
        with final_path.open("rb") as f:
            return pickle.load(f)

    def get_completed_trials(self, alg_idx: int, n_trials: int) -> list[int]:
        """
        Get list of completed trial numbers for an algorithm.

        Args:
            alg_idx: Algorithm index (0-based).
            n_trials: Total number of trials in the benchmark.

        Returns:
            List of completed trial numbers (0-based).

        """
        return [trial for trial in range(n_trials) if self.is_trial_complete(alg_idx, trial)]

    def load_metadata(self) -> dict[str, Any]:
        """
        Load checkpoint metadata.

        Returns:
            Dictionary containing benchmark_metadata and algorithms list.

        Raises:
            FileNotFoundError: If metadata.json doesn't exist.

        """
        metadata_path = self.checkpoint_dir / "metadata.json"
        with metadata_path.open(encoding="utf-8") as f:
            return json.load(f)

    def clear(self) -> None:
        """
        Remove entire checkpoint directory and all its contents.

        Warning:
            This permanently deletes all checkpoint data.

        """
        if self.checkpoint_dir.exists():
            shutil.rmtree(self.checkpoint_dir)
            LOGGER.info(f"Cleared checkpoint directory: {self.checkpoint_dir}")

    def _get_algorithm_dir(self, alg_idx: int) -> Path:
        """Get directory path for an algorithm."""
        return self.checkpoint_dir / f"algorithm_{alg_idx}"

    def _get_trial_dir(self, alg_idx: int, trial: int) -> Path:
        """Get directory path for a specific trial."""
        return self._get_algorithm_dir(alg_idx) / f"trial_{trial}"

    def _save_metadata(self, metadata: dict[str, Any]) -> None:
        """Save metadata to checkpoint directory."""
        metadata_path = self.checkpoint_dir / "metadata.json"
        with metadata_path.open("w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)
