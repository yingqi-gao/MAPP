"""MyersonNet model training and caching for experimental runs.

Train MyersonNet models with automatic caching to avoid redundant computation.
Caching is valuable because training MyersonNet models is expensive, especially
with large datasets and many iterations.
"""

from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch

from mapp.methods.deep_learning import (
    MyersonNet,
    train_myerson_net,
    save_myerson_net,
    load_myerson_net,
)
from mapp.utils.filename import generate_filename


def train_myerson_net_model(
    *,
    dist_name: str,
    train_bids: np.ndarray,
    train_seed: int,
    N_train: int,
    n_train: int,
    n_agents: int = 3,
    n_epochs: int = 50000,
    batch_size: int = 64,
    learning_rate: float = 0.001,
    force_retrain: bool = False,
) -> MyersonNet:
    """Train MyersonNet model with automatic caching.

    Slices training data to requested size, trains MyersonNet using gradient descent,
    and caches the result for future use. If a cached model exists with the same
    parameters, loads from cache instead of retraining.

    Args:
        dist_name: Distribution name for organizing cache
            ('truncnorm', 'truncexpon', 'beta', 'truncpareto', 'real')
        train_bids: Training auction data as 2D array (N_auctions × n_bids)
            Must have sufficient data: N_auctions ≥ N_train, n_bids ≥ n_train
        train_seed: Random seed used to generate training data
            (used for organizing cache files, not for training)
        N_train: Number of training auctions to use (sliced from train_bids)
        n_train: Number of training bids per auction to use (sliced from train_bids)
        n_agents: Number of agents/bidders in auction (default: 3)
        n_epochs: Number of training iterations (default: 50000)
        batch_size: Mini-batch size for training (default: 64)
        learning_rate: Adam optimizer learning rate (default: 0.001)
        force_retrain: If True, retrain even if cached model exists (default: False)

    Returns:
        Trained MyersonNet model (PyTorch model)

    Raises:
        ValueError: If train_bids is empty
        ValueError: If N_train > available auctions in train_bids
        ValueError: If n_train > available bids per auction in train_bids
        ValueError: If n_train not divisible by n_agents

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
        >>> # Train MyersonNet model using subset of data
        >>> model = train_myerson_net_model(
        ...     dist_name='truncnorm',
        ...     train_bids=train_bids,
        ...     train_seed=42,
        ...     N_train=50,
        ...     n_train=60,  # Must be divisible by n_agents=3
        ...     n_agents=3
        ... )

    Note:
        - Cache files are organized by training data source (auctions, bids, seed)
          and stored in mapp/workspace/{dist_name}/myerson_net_models/
        - Caching uses PyTorch format (.pt) for model state
        - n_train must be divisible by n_agents (bids reshaped into agent groups)
    """
    # Validate train data
    if train_bids is None or train_bids.size == 0:
        raise ValueError("No training data available for MyersonNet method")

    train_auctions, train_bids_per_auction = train_bids.shape

    if N_train > train_auctions:
        raise ValueError(
            f"Requested N_train={N_train} but only {train_auctions} training auctions available"
        )
    if n_train > train_bids_per_auction:
        raise ValueError(
            f"Requested n_train={n_train} but only {train_bids_per_auction} training bids per auction available"
        )
    if n_train % n_agents != 0:
        raise ValueError(
            f"n_train={n_train} must be divisible by n_agents={n_agents} for reshaping"
        )

    # Extract training data subset
    train_data = train_bids[:N_train, :n_train]

    # Reshape for multi-agent format: (N_train * samples_per_auction, n_agents)
    # Each auction contributes multiple agent groups
    samples_per_auction = n_train // n_agents
    total_samples = N_train * samples_per_auction

    # Reshape: (N_auctions, n_bids) → (total_samples, n_agents)
    train_data_reshaped = train_data.reshape(total_samples, n_agents)

    # Generate cache filepath
    filepath = generate_filename(
        dist_name=dist_name,
        file_type="myerson_net_models",
        subfolder=f"a{train_auctions}_b{train_bids_per_auction}_s{train_seed}",
        N=N_train,
        n=n_train,
        agents=n_agents,
        epochs=n_epochs,
    ).with_suffix(".pt")

    # Try to load from cache
    if not force_retrain and (cached_model := load_myerson_net(filepath)) is not None:
        print(f"   (train: {train_auctions}×{train_bids_per_auction}, used: {N_train}×{n_train}, agents: {n_agents})")
        return cached_model

    # Train new model
    print(f"🔄 Training MyersonNet (train: {train_auctions}×{train_bids_per_auction}, used: {N_train}×{n_train}, agents: {n_agents})...")
    model = train_myerson_net(
        train_bids=train_data_reshaped,
        n_agents=n_agents,
        n_epochs=n_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        verbose=True,
    )

    save_myerson_net(model, filepath)

    return model
