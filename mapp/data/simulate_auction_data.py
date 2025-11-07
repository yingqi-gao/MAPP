"""Simulate auction bid data for testing pricing methods."""

from functools import partial
from typing import Literal, Optional

import numpy as np

from mapp.core.auction import Auction
from mapp.data.auction_data_io import _load_experiment_data, _save_experiment_data
from mapp.data.generate_dist_params import _generate_dist_params
from mapp.methods.cdf_based.estimation.rbridge import _setup_rbridge
from mapp.utils.dist_loader import _get_scipy_distribution
from mapp.utils.filename import generate_filename


def _simulate_auction_batch(
    dist_name: str,
    n_auctions: int,
    bids_per_auction: int,
    seed: Optional[int],
    real_bids: Optional[list[np.ndarray]],
    lower: float,
    upper: float,
) -> list[Auction]:
    """Simulate auctions from distribution or bootstrap from real data."""
    rng = np.random.default_rng(seed)
    auctions = []

    if dist_name == 'real':
        if not real_bids:
            raise ValueError("real_bids required when dist_name='real'")
        _, _kde_cdf_r, _, _ = _setup_rbridge()

        for _ in range(n_auctions):
            source_idx = rng.integers(0, len(real_bids))
            source = real_bids[source_idx]
            bids = rng.choice(source, size=bids_per_auction, replace=True)
            cdf = _kde_cdf_r(bids, lower, upper)
            info = {"type": "bootstrap", "name": dist_name, "source_index": int(source_idx)}
            auctions.append(Auction(bids=bids, cdf=cdf, cdf_info=info))
    else:
        dist = _get_scipy_distribution(dist_name)
        for _ in range(n_auctions):
            params = _generate_dist_params(dist_name, rng=rng)
            bids = dist.rvs(size=bids_per_auction, random_state=rng, **params)
            cdf = partial(dist.cdf, **params)
            info = {"type": "known_dist", "name": dist_name, "params": params}
            auctions.append(Auction(bids=bids, cdf=cdf, cdf_info=info))

    return auctions


def simulate_experiment_data(
    *,
    dist_name: Literal['truncnorm', 'truncexpon', 'beta', 'truncpareto', 'real'],
    purpose: Literal['train', 'test'],
    n_auctions: int = 200,
    bids_per_auction: int = 200,
    seed: Optional[int] = None,
    n_runs: Optional[int] = None,
    force_regenerate: bool = False,
    real_bids: Optional[list[np.ndarray]] = None,
    lower: float = 1.0,
    upper: float = 10.0,
) -> list[list[Auction]]:
    """Generate experiment data with caching.

    real_bids: List of arrays. Single source: [array], mixed: [arr1, arr2, ...]
    """
    n_runs = 1 if purpose == 'train' else (n_runs or 100)
    if purpose == 'train' and n_runs != 1:
        raise ValueError("Training: n_runs must be 1")

    filepath = generate_filename(
        dist_name=dist_name, file_type="data",
        r=n_runs, a=n_auctions, b=bids_per_auction, s=seed, purpose=purpose
    )
    if not force_regenerate and (cached := _load_experiment_data(filepath)):
        return cached

    print(f"🔄 Generating {n_runs} runs for {dist_name}...")
    base_seed = seed or 0
    data = [
        _simulate_auction_batch(
            dist_name, n_auctions, bids_per_auction, base_seed + i,
            real_bids, lower, upper
        )
        for i in range(n_runs)
    ]

    # Prepare metadata for real data
    metadata = None
    if dist_name == 'real' and real_bids:
        metadata = {
            "real_bids": [arr.tolist() for arr in real_bids],
            "lower": lower,
            "upper": upper
        }

    _save_experiment_data(data, filepath, metadata=metadata)
    return data
