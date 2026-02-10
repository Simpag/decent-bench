"""Simple test to verify checkpoint functionality."""

import shutil
from pathlib import Path

import numpy as np

from decent_bench import benchmark, benchmark_problem
from decent_bench.costs import LinearRegressionCost
from decent_bench.distributed_algorithms import DGD

if __name__ == "__main__":
    # Create a simple test problem
    problem = benchmark_problem.create_regression_problem(
        cost_cls=LinearRegressionCost,
        n_agents=10,  # Use 10 agents to ensure n*d is even
        agent_state_snapshot_period=1,
    )

    # Test algorithm with short iterations
    iterations = 50
    algorithms = [
        DGD(iterations=iterations, step_size=0.01),
    ]

    checkpoint_dir = "./test_checkpoints"

    # Clean up if exists
    if Path(checkpoint_dir).exists():
        shutil.rmtree(checkpoint_dir)

    print("=" * 60)
    print("Testing checkpoint functionality")
    print("=" * 60)

    # Test 1: Run with checkpoints
    print("\n1. Running benchmark with checkpoints enabled...")
    try:
        benchmark.benchmark(
            algorithms=algorithms,
            benchmark_problem=problem,
            n_trials=2,
            checkpoint_dir=checkpoint_dir,
            checkpoint_step=10,  # Checkpoint every 10 iterations
            keep_n_checkpoints=2,
            max_processes=1,
            log_level=20,  # INFO level
        )
        print("Checkpoint test passed!")
    except Exception as e:
        print(f"Checkpoint test failed: {e}")
        raise

    # Verify checkpoint files were created
    checkpoint_path = Path(checkpoint_dir)
    print(f"\n2. Verifying checkpoint files...")
    print(f"   Checkpoint directory exists: {checkpoint_path.exists()}")
    print(f"   Initial network saved: {(checkpoint_path / 'initial_network.pkl').exists()}")
    print(f"   Metadata saved: {(checkpoint_path / 'metadata.json').exists()}")

    # List checkpoint files
    alg_dir = checkpoint_path / "algorithm_0"
    if alg_dir.exists():
        trial_dirs = list(alg_dir.iterdir())
        print(f"   Trial directories: {len(trial_dirs)}")
        for trial_dir in sorted(trial_dirs):
            checkpoints = list(trial_dir.glob("iter_*.pkl"))
            print(f"   - {trial_dir.name}: {len(checkpoints)} iteration checkpoints")

    # Test 2: Try to run again (should fail - directory not empty)
    print("\n3. Testing that benchmark() rejects non-empty checkpoint directory...")
    try:
        benchmark.benchmark(
            algorithms=algorithms,
            benchmark_problem=problem,
            n_trials=2,
            checkpoint_dir=checkpoint_dir,
            checkpoint_step=10,
            max_processes=1,
        )
        print("Test failed: should have raised ValueError!")
    except ValueError as e:
        if "not empty" in str(e):
            print(f"Correctly rejected non-empty directory")
        else:
            print(f"Wrong error: {e}")
            raise

    # Test 3: Test resume functionality
    print("\n4. Testing resume with resume_benchmarking=True...")
    print("   Note: Resume will load completed trials and skip re-running them")
    try:
        # This should work and recognize all trials are complete
        benchmark.benchmark(
            algorithms=algorithms,
            benchmark_problem=problem,
            n_trials=2,
            checkpoint_dir=checkpoint_dir,
            checkpoint_step=10,
            resume_benchmarking=True,  # Resume from existing checkpoint
            max_processes=1,
            log_level=20,
        )
        print("Resume test passed!")
    except Exception as e:
        print(f"Resume test failed: {e}")
        raise

    # Clean up
    print("\n5. Cleaning up...")
    shutil.rmtree(checkpoint_dir)
    print("Cleanup complete")

    print("\n" + "=" * 60)
    print("All tests completed!")
    print("=" * 60)
