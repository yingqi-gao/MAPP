"""Regrets I/O helpers for experiment results.

Internal module providing save/load functions for pickle-based regret results persistence.
Supports automatic checkpointing and resumability for long-running experiments.
"""

import pickle
from pathlib import Path
from typing import Any, Optional


def _save_regrets(results: dict[str, Any], filepath: Path) -> None:
    """Save experiment regret results to cache.

    Args:
        results: Results dictionary from run_experiments() containing regrets
        filepath: Path to save file

    Raises:
        RuntimeError: If save fails
    """
    filepath.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(filepath, "wb") as f:
            pickle.dump(results, f)
    except (IOError, pickle.PickleError) as e:
        raise RuntimeError(f"Failed to save regrets: {e}")


def _load_regrets(filepath: Path) -> Optional[dict[str, Any]]:
    """Load experiment regret results from cache.

    Args:
        filepath: Path to cached regret results file

    Returns:
        Results dictionary if file exists and loads successfully, None otherwise
    """
    if not filepath.exists():
        return None

    try:
        with open(filepath, "rb") as f:
            results = pickle.load(f)
        return results
    except Exception as e:
        print(f"⚠️  Failed to load cached regrets: {e}")
        return None
