"""Run pricing experiments for a single method.

This module orchestrates running a pricing method on pre-loaded test data:
1. Take experiment data (already loaded/generated)
2. Run pricing method on each auction
3. Collect and aggregate regrets per run
4. Return results for analysis
"""

import os
from concurrent.futures import ProcessPoolExecutor
from typing import Any, Literal, Optional

import numpy as np

from mapp.core.auction import Auction
from mapp.experiments.regrets_io import _save_regrets, _load_regrets
from mapp.utils.filename import generate_filename


def _process_single_run(args: tuple) -> tuple[int, float]:
    """Process a single experimental run (worker function for parallelization).

    Args:
        args: Tuple of (run_idx, auctions, method, k, n_auctions, bids_per_auction, method_kwargs)

    Returns:
        Tuple of (run_idx, avg_regret)
    """
    run_idx, auctions, method, k, n_auctions, bids_per_auction, method_kwargs = args

    # Run method on all auctions in this run
    auction_regrets = []
    for auction in auctions[:n_auctions]:
        # Slice bids if needed
        if len(auction.bids) > bids_per_auction:
            sliced_auction = Auction(
                bids=auction.bids[:bids_per_auction],
                cdf=auction.cdf,
                cdf_info=auction.cdf_info,
                ideal_price=auction._cached_ideal_price,
                ideal_revenue=auction._cached_ideal_revenue,
            )
        else:
            sliced_auction = auction

        _, _, regret = sliced_auction.run(method=method, k=k, **method_kwargs)
        auction_regrets.append(regret)

    return run_idx, float(np.mean(auction_regrets))


def _setup_checkpoint(
    dist_name: Optional[str],
    method: str,
    method_slug: Optional[str],
    bids_per_auction: int,
    k: int,
    n_runs: int,
):
    """Setup checkpoint path and try loading existing checkpoint.

    Returns:
        Tuple of (checkpoint_path, start_run, run_regrets, cached_result)
        - cached_result is the full result dict if already complete, None otherwise
    """
    checkpoint_path = None
    if dist_name:
        slug = method_slug if method_slug else method
        checkpoint_path = generate_filename(
            dist_name=dist_name,
            file_type="regrets",
            b=bids_per_auction,
            k=k,
            method=slug,
        ).with_suffix(".pkl")

    start_run = 0
    run_regrets = []

    if checkpoint_path and checkpoint_path.exists():
        cached = _load_regrets(checkpoint_path)
        if cached:
            runs_completed = cached.get('runs_completed', len(cached.get('regrets', [])))

            if runs_completed >= n_runs:
                print("✅ Experiment already complete (loaded from cache)")
                return checkpoint_path, 0, [], cached

            if runs_completed > 0:
                start_run = runs_completed
                run_regrets = cached['regrets']
                print(f"📂 Resuming from checkpoint: {start_run}/{n_runs} runs completed")

    return checkpoint_path, start_run, run_regrets, None


def run_experiments(
    *,
    experiment_data: list[list[Auction]],
    method: Literal["ideal", "ecdf", "kde", "rde", "myerson"],
    n_auctions: int,
    bids_per_auction: int,
    k: int = 2,
    n_jobs: int = 1,
    dist_name: Optional[str] = None,
    method_slug: Optional[str] = None,
    checkpoint_every: int = 100,
    **method_kwargs: Any,
) -> dict[str, Any]:
    """Run pricing experiments for a single method on pre-loaded data.

    Args:
        experiment_data: List of experimental runs, each containing a list of Auction objects
        method: Pricing method to evaluate ('ideal', 'ecdf', 'kde', 'rde', 'myerson')
        n_auctions: Number of auctions per experimental run
        bids_per_auction: Number of bids per auction
        k: Number of folds for group splitting (default: 2)
        n_jobs: Number of parallel jobs (1=sequential, -1=all cores, n>1=n cores)
        dist_name: Distribution name for checkpoint file (enables auto checkpointing)
        method_slug: Method slug for checkpoint filename (defaults to method name)
        checkpoint_every: Checkpoint every N runs (default: 100)
        **method_kwargs: Additional arguments for method (e.g., rde_model for RDE)

    Returns:
        Dictionary with:
            - method: Method name
            - n_runs: Number of runs
            - n_auctions: Number of auctions per run
            - bids_per_auction: Number of bids per auction
            - regrets: List of regrets per run
            - mean_regret: Mean regret across runs
            - std_regret: Standard deviation of regret across runs
    """
    n_runs = len(experiment_data)

    # Setup checkpoint and check if already complete
    checkpoint_path, start_run, run_regrets, cached = _setup_checkpoint(
        dist_name, method, method_slug, bids_per_auction, k, n_runs
    )
    if cached is not None:
        # Ensure cached result has mean_regret and std_regret
        if 'mean_regret' not in cached:
            cached['mean_regret'] = float(np.mean(cached['regrets']))
        if 'std_regret' not in cached:
            cached['std_regret'] = float(np.std(cached['regrets']))
        return cached

    # Print progress header
    n_cores = 1 if n_jobs == 1 else (n_jobs if n_jobs > 0 else os.cpu_count())
    print(f"\n{'='*60}")
    print(f"Running experiments: {method}")
    print(f"Config: {n_runs} runs × {n_auctions} auctions (starting from run {start_run})")
    print(f"Mode: {'Sequential' if n_jobs == 1 else f'Parallel ({n_cores} cores)'}")
    print(f"{'='*60}\n")

    # Run experiments in batches with ProcessPoolExecutor
    max_workers = None if n_jobs == -1 else n_jobs

    for batch_start in range(start_run, n_runs, checkpoint_every):
        batch_end = min(batch_start + checkpoint_every, n_runs)

        tasks = [
            (run_idx, experiment_data[run_idx], method, k, n_auctions, bids_per_auction, method_kwargs)
            for run_idx in range(batch_start, batch_end)
        ]

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            batch_results = list(executor.map(_process_single_run, tasks))

        batch_results.sort(key=lambda x: x[0])
        run_regrets.extend([regret for _, regret in batch_results])

        if checkpoint_path:
            _save_regrets({
                'method': method,
                'n_runs': n_runs,
                'n_auctions': n_auctions,
                'bids_per_auction': bids_per_auction,
                'regrets': run_regrets,
                'runs_completed': batch_end,
            }, checkpoint_path)

        print(f"📊 Progress: {batch_end}/{n_runs} runs complete")

    print("✅ All runs complete")

    # Calculate summary statistics
    mean_regret = float(np.mean(run_regrets))
    std_regret = float(np.std(run_regrets))

    # Print summary
    print(f"\n{'='*60}")
    print(f"Results for {method}")
    print(f"{'='*60}")
    print(f"Mean regret: {mean_regret:.4f}")
    print(f"Std regret:  {std_regret:.4f}")
    print(f"{'='*60}\n")

    return {
        'method': method,
        'n_runs': n_runs,
        'n_auctions': n_auctions,
        'bids_per_auction': bids_per_auction,
        'regrets': run_regrets,
        'mean_regret': mean_regret,
        'std_regret': std_regret,
    }
