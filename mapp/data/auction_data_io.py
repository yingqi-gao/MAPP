"""Auction data I/O helpers for experiments.

Internal module providing save/load functions for JSON-based persistence of
experiment data (list[list[Auction]]) used in pricing method comparisons.
"""

import json
from pathlib import Path
from typing import Optional

from mapp.core.auction import Auction


def _load_experiment_data(filepath: Path) -> Optional[list[list[Auction]]]:
    """Load experiment data from cache.

    Args:
        filepath: Path to cached file

    Returns:
        List of experiment data if file exists and loads successfully, None otherwise
    """
    if not filepath.exists():
        return None

    try:
        print(f"📂 Loading cached data: {filepath.name}")
        with open(filepath, "r") as f:
            cached = json.load(f)

        # Handle new format with metadata or old format (backward compatible)
        metadata = None
        precomputed_cdfs = None
        if isinstance(cached, dict) and "runs" in cached:
            # New format: {"metadata": {...}, "runs": [[...], [...]]}
            metadata = cached.get("metadata")
            serialized = cached["runs"]

            # Pre-compute CDFs for bootstrap data
            if metadata and "real_bids" in metadata:
                import numpy as np
                from mapp.methods.cdf_based.estimation.rbridge import _setup_rbridge
                from mapp.utils.constants import VALUE_LOWER_BOUND, VALUE_UPPER_BOUND

                real_bids = metadata["real_bids"]
                lower = metadata.get("lower", VALUE_LOWER_BOUND)
                upper = metadata.get("upper", VALUE_UPPER_BOUND)

                _, _kde_cdf_r, _, _ = _setup_rbridge()
                precomputed_cdfs = [
                    _kde_cdf_r(np.array(source), lower, upper)
                    for source in real_bids
                ]
        else:
            # Old format: [[...], [...]]
            serialized = cached

        # Load auctions, passing metadata and precomputed CDFs
        experiment_data = [[Auction.from_dict(data, metadata=metadata, precomputed_cdfs=precomputed_cdfs) for data in run] for run in serialized]
        n_runs_loaded = len(experiment_data)
        n_auctions_loaded = len(experiment_data[0]) if experiment_data else 0
        print(f"✅ Loaded {n_runs_loaded} runs × {n_auctions_loaded} auctions")
        return experiment_data
    except Exception as e:
        raise RuntimeError(f"Failed to load experiment data: {e}")


def _save_experiment_data(
    experiment_data: list[list[Auction]],
    filepath: Path,
    metadata: Optional[dict] = None
) -> None:
    """Save experiment data to cache.

    Args:
        experiment_data: List of experiment data to save
        filepath: Path to save file
        metadata: Optional metadata dict (e.g., {"real_bids": [...], "lower": 1.0, ...})

    Raises:
        RuntimeError: If save fails
    """
    filepath.parent.mkdir(parents=True, exist_ok=True)
    try:
        serialized_runs = [[auction.to_dict() for auction in run] for run in experiment_data]

        # Save with metadata if provided, otherwise use old format for backward compatibility
        if metadata:
            output = {"metadata": metadata, "runs": serialized_runs}
        else:
            output = serialized_runs

        with open(filepath, "w") as f:
            json.dump(output, f, indent=2)
        file_size = filepath.stat().st_size
        print(f"✅ Saved {len(experiment_data)} runs ({file_size:,} bytes)")
    except (IOError, TypeError) as e:
        raise RuntimeError(f"Failed to save experiment data: {e}")
