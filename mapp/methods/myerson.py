"""
Empirical Myerson auction implementation following Cole and Roughgarden (2014).

This module implements the empirical Myerson mechanism for revenue optimization
in auctions. The key insight is to use historical bid data to estimate virtual
valuation functions and compute optimal reserve prices.

Reference:
    Cole, R., & Roughgarden, T. (2014). The sample complexity of revenue maximization.
    In Proceedings of the forty-sixth annual ACM symposium on Theory of computing (pp. 243-252).
"""

from typing import Optional

import numpy as np
from scipy.spatial import ConvexHull


def _create_empirical_quantile_coordinates(sorted_bids: np.ndarray) -> np.ndarray:
    """
    Construct empirical quantile coordinates for Myerson mechanism.

    For each bid value v_j, creates point ((2j+1)/(2m), (2j+1)*v_j/(2m))
    where j is the rank and m is the total number of bids.

    Args:
        sorted_bids: Bid values sorted in descending order

    Returns:
        Array of shape (m, 2) with quantile coordinates
    """
    num_bids = len(sorted_bids)
    coordinates = np.zeros((num_bids, 2))

    for rank in range(num_bids):
        quantile = (2 * rank + 1) / (2 * num_bids)
        coordinates[rank, 0] = quantile  # x-coordinate: empirical quantile
        coordinates[rank, 1] = quantile * sorted_bids[rank]  # y-coordinate: scaled value

    return coordinates


def _get_virtual_value_slope(quantile: float, hull_points: np.ndarray, segment_slopes: list[float]) -> Optional[float]:
    """
    Calculate the empirical ironed virtual value (slope) for a given quantile.

    The virtual value is the slope of the convex hull at the given quantile.
    Special handling for boundary points to ensure monotonicity.

    Args:
        quantile: The quantile value (x-coordinate) to evaluate
        hull_points: Vertices of the convex hull, sorted by x-coordinate
        segment_slopes: Slopes of each segment of the convex hull

    Returns:
        The virtual value (slope) at the given quantile, or None if not found
    """
    for i in range(len(hull_points) - 1):
        x_left, x_right = hull_points[i][0], hull_points[i + 1][0]

        if x_left <= quantile <= x_right:
            # Handle boundary conditions to maintain monotonicity
            if quantile == x_right and i < len(segment_slopes) - 1:
                return max(segment_slopes[i], segment_slopes[i + 1])
            if quantile == x_left and i > 0:
                return max(segment_slopes[i - 1], segment_slopes[i])
            return segment_slopes[i]

    return None


def _find_optimal_reserve_index(virtual_values: list[float]) -> int:
    """
    Find the index corresponding to the optimal reserve price.

    The optimal reserve price corresponds to the bid with the smallest
    non-negative virtual value (ironed virtual valuation).

    Args:
        virtual_values: list of virtual values for each bid

    Returns:
        Index of the bid that should be used as reserve price

    Raises:
        ValueError: If no valid reserve price index is found
    """
    min_non_negative_value = float('inf')
    optimal_index = -1

    for i, virtual_value in enumerate(virtual_values):
        if virtual_value >= 0 and virtual_value <= min_non_negative_value:
            min_non_negative_value = virtual_value
            optimal_index = i

    if optimal_index == -1:
        raise ValueError("No valid reserve price found: all virtual values are negative")

    return optimal_index


def compute_myerson_reserve_price(bids: np.ndarray) -> float:
    """
    Compute optimal reserve price using the empirical Myerson mechanism.

    Implementation of Cole & Roughgarden (2014) algorithm:
    1. Create empirical quantile representation of bid distribution
    2. Compute convex hull to get "ironed" virtual valuation function
    3. Find optimal reserve as bid with smallest non-negative virtual value

    Args:
        bids: Historical bid data as numpy array

    Returns:
        Optimal reserve price

    Raises:
        ValueError: If input is empty or no valid reserve price exists

    Example:
        >>> bids = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        >>> reserve = compute_myerson_reserve_price(bids)
        >>> print(f"Optimal reserve: {reserve}")
    """
    if len(bids) == 0:
        raise ValueError("Cannot compute reserve price from empty bid array")

    # Step 1: Sort bids in descending order and create quantile coordinates
    sorted_bids = np.sort(bids)[::-1]
    quantile_coords = _create_empirical_quantile_coordinates(sorted_bids)

    # Step 2: Add boundary points for convex hull computation
    # These represent the "no sale" outcomes at quantiles 0 and 1
    boundary_points = np.array([[0.0, 0.0], [1.0, 0.0]])
    all_points = np.concatenate(([boundary_points[0]], quantile_coords, [boundary_points[1]]), axis=0)

    # Step 3: Compute convex hull to get "ironed" virtual valuation function
    try:
        hull = ConvexHull(all_points)
        hull_vertices = all_points[hull.vertices]
        # Sort hull points by x-coordinate (quantile)
        hull_vertices_sorted = hull_vertices[hull_vertices[:, 0].argsort()]
    except Exception as e:
        raise ValueError(f"Failed to compute convex hull: {e}")

    # Step 4: Calculate slopes (virtual values) for each hull segment
    segment_slopes = []
    for i in range(len(hull_vertices_sorted) - 1):
        x1, y1 = hull_vertices_sorted[i]
        x2, y2 = hull_vertices_sorted[i + 1]

        if x2 == x1:  # Avoid division by zero
            slope = 0.0
        else:
            slope = (y2 - y1) / (x2 - x1)
        segment_slopes.append(slope)

    # Step 5: Compute virtual value for each original point
    virtual_values = []
    for point in all_points:
        quantile = point[0]
        virtual_value = _get_virtual_value_slope(quantile, hull_vertices_sorted, segment_slopes)
        if virtual_value is None:
            raise ValueError(f"Could not compute virtual value for quantile {quantile}")
        virtual_values.append(virtual_value)

    # Step 6: Find optimal reserve (exclude boundary points)
    bid_virtual_values = virtual_values[1:-1]  # Remove boundary points
    optimal_index = _find_optimal_reserve_index(bid_virtual_values)
    return float(sorted_bids[optimal_index])


def myerson_pricing(bids: np.ndarray) -> float:
    """
    Myerson auction pricing method using bids.

    Uses the empirical Myerson mechanism to compute optimal reserve prices
    based on historical bid data. This method implements the Cole & Roughgarden
    algorithm for revenue maximization in single-item auctions.

    Args:
        bids: Bid array to analyze

    Returns:
        Optimal reserve price (float)

    Example:
        >>> bids = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        >>> price = myerson_pricing(bids)
        >>> print(f"Optimal reserve: {price:.2f}")
    """
    if len(bids) == 0:
        return 0.0

    # Compute Myerson optimal reserve price
    reserve_price = compute_myerson_reserve_price(bids)

    return float(reserve_price)
