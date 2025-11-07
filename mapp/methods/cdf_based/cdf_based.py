"""CDF-based pricing methods for auction optimization.

This module provides pricing strategies that estimate cumulative distribution
functions (CDFs) from bid data and optimize posted prices to maximize revenue.

Main functions:
    - cdf_based_pricing: Estimate CDF from bids and find optimal price
    - optimize_price: Find optimal price given a known CDF

Supported CDF estimation methods:
    - ecdf: Empirical CDF (non-parametric)
    - kde: Kernel Density Estimation (smooth)
    - rde: Repeated Density Estimation (FPCA-based, requires training data)
"""

import warnings
from typing import Callable, cast

import numpy as np
from scipy.optimize import OptimizeWarning, OptimizeResult, minimize_scalar

from mapp.methods.cdf_based.estimation.rbridge import _setup_rbridge
from mapp.utils.constants import (
    OPTIMIZATION_BUFFER_RATIO,
    OPTIMIZATION_MAX_ITER,
    OPTIMIZATION_TOLERANCE,
    _get_optimization_bounds,
)
from mapp.utils.revenue import _calculate_expected_revenue

# Initialize R bridge and load CDF estimation functions
_ecdf_r, _kde_cdf_r, _train_rde_r, _rde_cdf_r = _setup_rbridge()


def cdf_based_pricing(
    *,
    bids: np.ndarray,
    method: str = "ecdf",
    train_bids: np.ndarray | None = None,
    rde_model: dict | None = None,
) -> float:
    """CDF-based pricing using estimation methods.

    Args:
        bids: Bid array to analyze
        method: CDF estimation method ("ecdf", "kde", "rde")
        train_bids: Training data for RDE method (2D numpy array: n_auctions × n_bids_per_auction)
        rde_model: Pre-trained RDE model. If provided, skips training and uses this model directly.

    Returns:
        Optimal price

    Note:
        Optimization bounds are configured in mapp.utils.constants
    """
    # Get bounds from constants or data
    lower, upper = _get_optimization_bounds(bids)

    # Get estimated CDF based on method
    estimated_cdf = _get_cdf(method, bids, train_bids, rde_model, lower, upper)

    # Optimize price using estimated CDF
    return optimize_price(estimated_cdf, lower, upper)


def optimize_price(cdf: Callable[[float], float], lower: float, upper: float) -> float:
    """Optimize price given a CDF function.

    Args:
        cdf: Cumulative distribution function
        lower: Lower bound for optimization
        upper: Upper bound for optimization

    Returns:
        Optimal price that maximizes expected revenue under the given CDF
    """
    # Add small buffer to bounds to avoid numerical issues at boundaries
    buffer = (upper - lower) * OPTIMIZATION_BUFFER_RATIO
    bounds = (lower + buffer, upper - buffer)

    # Optimize to find price that maximizes expected revenue
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", OptimizeWarning)
        res = cast(OptimizeResult, minimize_scalar(
            lambda p: -_calculate_expected_revenue(p, cdf),
            bounds=bounds,
            method="bounded",
            options={"xatol": OPTIMIZATION_TOLERANCE, "maxiter": OPTIMIZATION_MAX_ITER},
        ))

    if res.success:
        return float(res.x)

    raise RuntimeError("Price optimization failed")


def _get_cdf(
    method: str,
    bids: np.ndarray,
    train_bids: np.ndarray | None,
    rde_model: dict | None,
    lower: float,
    upper: float,
) -> Callable[[float], float]:
    """Get CDF function based on estimation method.

    Args:
        method: Estimation method ("ecdf", "kde", or "rde")
        bids: Bid observations to estimate CDF from
        train_bids: Training data for RDE (required if rde_model not provided)
        rde_model: Pre-trained RDE model (optional, skips training if provided)
        lower: Lower bound for grid-based methods (kde, rde)
        upper: Upper bound for grid-based methods (kde, rde)

    Returns:
        Callable CDF function

    Raises:
        ValueError: If method is unknown or RDE requirements not met
    """
    if method == "ecdf":
        return _ecdf_r(bids)

    elif method == "kde":
        return _kde_cdf_r(bids, lower, upper)

    elif method == "rde":
        if rde_model is not None:
            # Use pre-trained model
            return _rde_cdf_r(bids, rde_model)
        elif train_bids is not None:
            # Train new model
            trained_model = _train_rde_r(train_bids, lower, upper)
            return _rde_cdf_r(bids, trained_model)
        else:
            raise ValueError(
                "RDE method requires either a pre-trained model (rde_model) "
                "or training data (train_bids). Neither was provided."
            )

    else:
        raise ValueError(f"Unknown method: {method}. Use 'ecdf', 'kde', or 'rde'")
