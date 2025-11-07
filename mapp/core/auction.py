"""Auction management for pricing methods.

This module provides the Auction class for running pricing experiments
with group-based strategies and regret analysis.
"""

from dataclasses import dataclass
from typing import Callable, Literal, Optional
from functools import partial

import numpy as np

from mapp.methods.cdf_based.cdf_based import cdf_based_pricing, optimize_price
from mapp.methods.cdf_based.estimation.rbridge import _setup_rbridge
from mapp.methods.myerson import myerson_pricing
from mapp.methods.deep_learning import myerson_net_pricing
from mapp.utils.constants import VALUE_LOWER_BOUND, VALUE_UPPER_BOUND
from mapp.utils.dist_loader import _get_scipy_distribution
from mapp.utils.revenue import _calculate_expected_revenue

# Initialize R bridge for ECDF
_ecdf_r, _, _, _ = _setup_rbridge()

@dataclass
class Auction:
    """Single auction with bids and true value CDF."""

    bids: np.ndarray
    cdf: Callable[[float], float]
    cdf_info: dict
    ideal_price: Optional[float] = None
    ideal_revenue: Optional[float] = None

    def __post_init__(self):
        assert len(self.bids) > 0 and callable(self.cdf), "Invalid auction inputs"

        # Calculate and cache ideal (oracle) price and revenue if not provided
        if self.ideal_price is None or self.ideal_revenue is None:
            ideal_price = optimize_price(
                self.cdf, VALUE_LOWER_BOUND, VALUE_UPPER_BOUND
            )  # Ideal method always uses fixed value bounds
            ideal_revenue = self._calculate_revenue(ideal_price)
            self._cached_ideal_price = ideal_price
            self._cached_ideal_revenue = ideal_revenue
        else:
            self._cached_ideal_price = self.ideal_price
            self._cached_ideal_revenue = self.ideal_revenue

    def run(
        self,
        *,
        method: Literal["ideal", "ecdf", "kde", "rde", "myerson"] = "ideal",
        k: int = 2,
        seed: Optional[int] = None,
        **kwargs
    ) -> tuple[float, float, float]:
        """Run auction with complete workflow: price → revenue → regret.

        Args:
            method: Pricing method
                    - Oracle: "ideal"
                    - CDF-based: "ecdf", "kde", "rde"
                    - Myerson: "myerson"
            k: Number of folds for group splitting (default: 2, not used for ideal)
            seed: Random seed for splitting
            **kwargs: Additional arguments for pricing method

        Returns:
            Tuple of (price, revenue, regret)
        """
        # Special case: ideal method returns cached results
        if method == "ideal":
            return self._cached_ideal_price, self._cached_ideal_revenue, 0.0

        # Run leave-one-group-out pricing to get max price
        price = self._run_logo_pricing(method, k, seed, **kwargs)

        # Calculate revenue and regret
        revenue = self._calculate_revenue(price)
        regret = self._cached_ideal_revenue - revenue

        return price, revenue, regret

    def _calculate_revenue(self, price: float) -> float:
        """Calculate expected revenue for a given price.

        Args:
            price: Posted price

        Returns:
            Expected revenue using the auction's CDF
        """
        return _calculate_expected_revenue(price, self.cdf)

    def _run_logo_pricing(self, method: str, k: int, seed: Optional[int], **kwargs) -> float:
        """Run leave-one-group-out pricing and return maximum price.

        Splits bids into k groups, then for each group, trains on all other groups
        to get a price. Returns the maximum of these k prices.

        Args:
            method: Pricing method to use
            k: Number of groups
            seed: Random seed for splitting
            **kwargs: Additional arguments for pricing method

        Returns:
            Maximum price across all k leave-one-group-out iterations

        Raises:
            ValueError: If method is unknown or no valid groups generated
        """
        groups = self._split_bids(k, seed)
        group_prices = []

        for group_idx in range(k):
            # Get training indices (all groups except current one)
            train_indices = np.concatenate([g for i, g in enumerate(groups) if i != group_idx])

            # Get price for this group
            train_bids = self.bids[train_indices]
            price = self._get_price_for_method(method, train_bids, **kwargs)
            group_prices.append(price)

        if not group_prices:
            raise ValueError("No valid training groups generated")

        return float(np.max(group_prices))

    def _split_bids(self, k: int, seed: Optional[int] = None) -> list[np.ndarray]:
        """Create k-fold indices splits for group-based pricing.

        Args:
            k: Number of groups
            seed: Random seed for reproducibility

        Returns:
            List of k arrays, each containing indices for one group

        Note:
            For leave-one-out (k == n_bids), returns sequential splits without shuffling.
            For standard k-fold, shuffles indices before splitting.
        """
        n_bids = len(self.bids)
        assert k > 0 and k <= n_bids, f"k={k} must be in range [1, {n_bids}]"
        rng = np.random.default_rng(seed)

        # Shuffle indices (or use sequential for leave-one-out)
        indices = np.arange(n_bids) if k == n_bids else rng.permutation(n_bids)

        # Split into k groups (handles unequal sizes automatically)
        return np.array_split(indices, k)
    
    def _get_price_for_method(self, method: str, bids: np.ndarray, **kwargs) -> float:
        """Get price using the specified pricing method.

        Args:
            method: Pricing method to use
            bids: Bid data to price
            **kwargs: Additional arguments for pricing method

        Returns:
            Optimal price

        Raises:
            ValueError: If method is unknown
        """
        if method in ["ecdf", "kde", "rde"]:
            return cdf_based_pricing(bids=bids, method=method, **kwargs)
        elif method == "myerson":
            return myerson_pricing(bids)
        elif method == "myerson_net":
            return myerson_net_pricing(bids=bids, **kwargs)
        else:
            raise ValueError(f"Unknown method '{method}'. Valid: ideal, ecdf, kde, rde, myerson, myerson_net")

    def to_dict(self) -> dict:
        """Serialize auction to dictionary.

        Returns:
            Dictionary with bids, cdf_info, and cached ideal values
        """
        return {
            "bids": self.bids.tolist(),
            "cdf_info": self.cdf_info,
            "ideal_price": self._cached_ideal_price,
            "ideal_revenue": self._cached_ideal_revenue,
        }

    @classmethod
    def from_dict(cls, data: dict, metadata: Optional[dict] = None, precomputed_cdfs: Optional[list] = None) -> 'Auction':
        """Deserialize auction from dictionary.

        Args:
            data: Dictionary with bids, cdf_info, and ideal values
            metadata: Optional metadata dict containing real_bids, lower, upper
            precomputed_cdfs: Optional list of pre-computed CDFs for bootstrap data

        Returns:
            Reconstructed Auction object

        Raises:
            ValueError: If cdf_info type is unknown
        """
        bids = np.array(data["bids"])
        cdf_info = data["cdf_info"]
        ideal_price = data.get("ideal_price")
        ideal_revenue = data.get("ideal_revenue")

        # Reconstruct CDF based on type
        cdf_type = cdf_info.get("type")
        if cdf_type == "known_dist":
            dist = _get_scipy_distribution(cdf_info["name"])
            cdf = partial(dist.cdf, **cdf_info["params"])
        elif cdf_type == "ecdf":
            source_bids = np.array(cdf_info["source_bids"])
            cdf = _ecdf_r(source_bids)  # Use R's ecdf function
        elif cdf_type == "bootstrap":
            # Use pre-computed CDF from the list
            if not precomputed_cdfs:
                raise ValueError("Bootstrap type requires precomputed_cdfs")

            source_index = cdf_info["source_index"]
            cdf = precomputed_cdfs[source_index]
        else:
            raise ValueError(f"Unknown cdf_info type: '{cdf_type}'. Valid: known_dist, ecdf, bootstrap")

        return cls(
            bids=bids,
            cdf=cdf,
            cdf_info=cdf_info,
            ideal_price=ideal_price,
            ideal_revenue=ideal_revenue,
        )
