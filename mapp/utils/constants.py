"""Global configuration for MAPP experiments."""

import numpy as np

# Value bounds for bid generation
VALUE_LOWER_BOUND: float = 1.0
VALUE_UPPER_BOUND: float = 10.0

# Optimization bounds for pricing methods
# - Set to float values (e.g., 1.0, 10.0) to use fixed bounds
# - Set to None to use data-dependent bounds (min/max of bids)
OPTIMIZATION_LOWER_BOUND: float | None = 1.0
OPTIMIZATION_UPPER_BOUND: float | None = 10.0

# Optimization algorithm parameters
OPTIMIZATION_BUFFER_RATIO: float = 0.001  # Buffer to avoid boundary issues
OPTIMIZATION_TOLERANCE: float = 1e-8      # Convergence tolerance (xatol)
OPTIMIZATION_MAX_ITER: int = 500          # Maximum optimization iterations


def _get_optimization_bounds(bids: np.ndarray) -> tuple[float, float]:
    """Get optimization bounds from constants or data.

    Internal utility function.

    Uses OPTIMIZATION_LOWER_BOUND and OPTIMIZATION_UPPER_BOUND from constants.
    If these are set to None, falls back to min/max of bids.

    Args:
        bids: Bid array to derive bounds from if constants are None

    Returns:
        Tuple of (lower, upper) bounds for optimization
    """
    lower = OPTIMIZATION_LOWER_BOUND if OPTIMIZATION_LOWER_BOUND is not None else float(np.min(bids))
    upper = OPTIMIZATION_UPPER_BOUND if OPTIMIZATION_UPPER_BOUND is not None else float(np.max(bids))
    return lower, upper
