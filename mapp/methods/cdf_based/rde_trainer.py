"""RDE model training and caching for experimental runs.

Train RDE models with automatic caching to avoid redundant computation.
Caching is valuable because training RDE models is expensive, especially with large datasets.
"""

from typing import Any

import numpy as np

from mapp.methods.cdf_based.estimation.rbridge import _setup_rbridge
from mapp.methods.cdf_based.rde_model_io import _load_rde_model, _save_rde_model
from mapp.utils.constants import _get_optimization_bounds
from mapp.utils.filename import generate_filename


def train_rde_model(
    *,
    dist_name: str,
    train_bids: np.ndarray,
    train_seed: int,
    N_train: int,
    n_train: int,
    force_retrain: bool = False,
) -> Any:
    """Train RDE model with automatic caching.

    Slices training data to requested size, trains RDE model using FPCA,
    and caches the result for future use. If a cached model exists with
    the same parameters, loads from cache instead of retraining.

    Args:
        dist_name: Distribution name for organizing cache
            ('truncnorm', 'truncexpon', 'beta', 'truncpareto', 'real')
        train_bids: Training auction data as 2D array (N_auctions × n_bids)
            Must have sufficient data: N_auctions ≥ N_train, n_bids ≥ n_train
        train_seed: Random seed used to generate training data
            (used for organizing cache files, not for training)
        N_train: Number of training auctions to use (sliced from train_bids)
        n_train: Number of training bids per auction to use (sliced from train_bids)
        force_retrain: If True, retrain even if cached model exists (default: False)

    Returns:
        Trained RDE model object (R object via rpy2)

    Raises:
        ValueError: If train_bids is empty
        ValueError: If N_train > available auctions in train_bids
        ValueError: If n_train > available bids per auction in train_bids

    Example:
        >>> import numpy as np
        >>> from mapp.data.simulate_auction_data import simulate_experiment_data
        >>>
        >>> # Generate training data
        >>> train_data = simulate_experiment_data(
        ...     dist_name='truncnorm', purpose='train',
        ...     n_auctions=200, bids_per_auction=200, seed=42
        ... )
        >>> train_bids = np.array([auction.bids for auction in train_data[0]])
        >>>
        >>> # Train RDE model using subset of data
        >>> model = train_rde_model(
        ...     dist_name='truncnorm',
        ...     train_bids=train_bids,
        ...     train_seed=42,
        ...     N_train=50,
        ...     n_train=200
        ... )

    Note:
        - Optimization bounds are auto-derived from data or configured globally
          in mapp.utils.constants (OPTIMIZATION_LOWER_BOUND, OPTIMIZATION_UPPER_BOUND)
        - Cache files are organized by training data source (auctions, bids, seed)
          and stored in mapp/workspace/{dist_name}/rde_models/
        - Caching uses pickle format (.pkl) for R objects via rpy2
    """
    # Validate train data
    if train_bids is None or train_bids.size == 0:
        raise ValueError("No training data available for RDE method")

    train_auctions, train_bids_per_auction = train_bids.shape

    if N_train > train_auctions:
        raise ValueError(
            f"Requested N_train={N_train} but only {train_auctions} training auctions available"
        )
    if n_train > train_bids_per_auction:
        raise ValueError(
            f"Requested n_train={n_train} but only {train_bids_per_auction} training bids per auction available"
        )

    # Extract training data subset and get bounds from constants
    train_data = train_bids[:N_train, :n_train]
    lower, upper = _get_optimization_bounds(train_data.flatten())

    # Generate cache filepath
    filepath = generate_filename(
        dist_name=dist_name,
        file_type="rde_models",
        subfolder=f"a{train_auctions}_b{train_bids_per_auction}_s{train_seed}",
        N=N_train,
        n=n_train,
        l=lower,
        u=upper,
    ).with_suffix(".pkl")

    # Try to load from cache
    if not force_retrain and (cached_model := _load_rde_model(filepath)) is not None:
        print(f"   (train: {train_auctions}×{train_bids_per_auction}, used: {N_train}×{n_train})")
        return cached_model

    # Train new model
    print(f"🔄 Training RDE model (train: {train_auctions}×{train_bids_per_auction}, used: {N_train}×{n_train})...")
    _, _, _train_rde_r, _ = _setup_rbridge()
    rde_model = _train_rde_r(train_data, lower, upper)
    _save_rde_model(rde_model, filepath)

    return rde_model
