"""RDE model I/O helpers.

Internal module providing save/load functions for pickle-based RDE model persistence.
"""

import pickle
from pathlib import Path
from typing import Any, Optional


def _load_rde_model(filepath: Path) -> Optional[Any]:
    """Load RDE model from cache.

    Args:
        filepath: Path to cached model file

    Returns:
        Trained RDE model if file exists and loads successfully, None otherwise
    """
    if not filepath.exists():
        return None

    try:
        print(f"📂 Loading cached RDE model: {filepath.parent.name}/{filepath.name}")
        with open(filepath, "rb") as f:
            rde_model = pickle.load(f)
        print("✅ Loaded RDE model")
        return rde_model
    except Exception as e:
        print(f"⚠️  Failed to load cached model: {e}")
        print("   Retraining model...")
        return None


def _save_rde_model(rde_model: Any, filepath: Path) -> None:
    """Save RDE model to cache.

    Args:
        rde_model: Trained RDE model to save
        filepath: Path to save file

    Raises:
        RuntimeError: If save fails (non-fatal, just warns)
    """
    filepath.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(filepath, "wb") as f:
            pickle.dump(rde_model, f)
        file_size = filepath.stat().st_size
        print(f"✅ Saved RDE model ({file_size:,} bytes)")
    except Exception as e:
        print(f"⚠️  Failed to save RDE model: {e}")
        print("   Model will still be used but not cached")
