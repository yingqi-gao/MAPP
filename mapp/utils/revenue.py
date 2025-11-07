"""Revenue calculation utilities for auction pricing.

Internal module - provides helper functions for revenue calculations.
"""

from typing import Callable
import numpy as np


def _calculate_expected_revenue(price: float, cdf: Callable) -> float:
    """Calculate expected revenue for a given price and CDF.

    Internal utility function.

    Args:
        price: Posted price
        cdf: Cumulative distribution function

    Returns:
        Expected revenue: price * (1 - CDF(price))
    """
    prob_sale = 1.0 - float(np.clip(cdf(price), 0.0, 1.0))
    return price * prob_sale