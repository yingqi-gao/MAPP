"""Distribution utilities for working with scipy.stats distributions.

Internal module - provides helper functions for loading scipy distributions.
"""

import scipy.stats


def _get_scipy_distribution(dist_name: str):
    """Get scipy.stats distribution by name.

    Internal utility function.

    Args:
        dist_name: Distribution name (e.g., 'truncnorm', 'beta', 'truncexpon')

    Returns:
        scipy.stats distribution object

    Raises:
        ValueError: If distribution not found in scipy.stats
    """
    try:
        return getattr(scipy.stats, dist_name)
    except AttributeError:
        raise ValueError(f"Distribution '{dist_name}' not found in scipy.stats")
