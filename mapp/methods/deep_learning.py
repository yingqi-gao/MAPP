"""Deep learning-based auction pricing using MyersonNet.

This module implements MyersonNet, a neural network that learns optimal
single-item auction mechanisms by training on bid distributions.

Based on "Optimal Auctions through Deep Learning" (Dütting et al., 2019)
Modernized implementation using PyTorch for Python 3.11+
"""

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class MyersonNet(nn.Module):
    """Neural network for learning optimal auction mechanisms.

    Learns to compute virtual valuations and allocations for single-item auctions
    with multiple bidders, maximizing seller revenue while maintaining incentive
    compatibility.

    Architecture:
        - Encoding layer: Linear transformation to virtual values
        - Allocation: Softmax over virtual values (smooth approximation during training)
        - Payment: Second-price logic based on virtual values
    """

    def __init__(
        self,
        n_agents: int = 3,
        n_linear_funcs: int = 10,
        n_max_units: int = 10,
        temperature: float = 1000.0,
    ):
        """Initialize MyersonNet.

        Args:
            n_agents: Number of bidders in the auction
            n_linear_funcs: Number of linear functions for encoding
            n_max_units: Number of max units in the network
            temperature: Softmax temperature for smooth allocation (higher = closer to argmax)
        """
        super().__init__()
        self.n_agents = n_agents
        self.n_linear_funcs = n_linear_funcs
        self.n_max_units = n_max_units
        self.temperature = temperature

        # Encoding weights for virtual value transformation
        # w1: multiplicative weights, w2: bias weights (scaled by -5.0 as in original)
        self.w1 = nn.Parameter(torch.randn(n_max_units, n_linear_funcs, n_agents))
        self.w2 = nn.Parameter(-torch.rand(n_max_units, n_linear_funcs, n_agents) * 5.0)

    def forward(self, bids: torch.Tensor, training: bool = True) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass: compute allocations and payments.

        Args:
            bids: Bid tensor of shape (batch_size, n_agents)
            training: If True, use soft allocation; if False, use hard allocation

        Returns:
            Tuple of (allocations, payments, revenue):
                - allocations: (batch_size, n_agents) - probability each agent wins
                - payments: (batch_size, n_agents) - payment for each agent
                - revenue: scalar - expected revenue across batch
        """
        batch_size = bids.shape[0]

        # Expand bids for broadcasting: (batch, agents) -> (batch, max_units, linear_funcs, agents)
        bids_expanded = bids.unsqueeze(1).unsqueeze(2)  # (batch, 1, 1, agents)

        # Compute virtual values using learned transformation
        # Formula: min_u max_l (bid_i * exp(w1) + w2)
        virtual_interim = bids_expanded * torch.exp(self.w1) + self.w2  # (batch, max_units, linear_funcs, agents)
        virtual_max = torch.max(virtual_interim, dim=2)[0]  # (batch, max_units, agents)
        virtual_values = torch.min(virtual_max, dim=1)[0]  # (batch, agents)

        # Allocation: Append dummy agent (reserve price 0) then softmax/argmax
        # Matches original TF implementation lines 78-84
        zeros = torch.zeros(batch_size, 1, device=bids.device)
        vv_with_dummy = torch.cat([virtual_values, zeros], dim=1)  # (batch, agents+1)

        if training:
            # Soft allocation: softmax with high temperature (1000) to approximate argmax
            # Original: softmax(vv @ append_dummy_mat @ diag(1000))
            # Simplified: softmax(vv * 1000) since diag just scales
            a_dummy = torch.softmax(vv_with_dummy * self.temperature, dim=1)
        else:
            # Hard allocation: argmax (winner-take-all)
            winner = torch.argmax(vv_with_dummy, dim=1)
            a_dummy = torch.zeros_like(vv_with_dummy)
            a_dummy.scatter_(1, winner.unsqueeze(1), 1.0)

        # Remove dummy agent column to get final allocations
        allocations = a_dummy[:, :self.n_agents]

        # Payment: Second-price auction on virtual values
        # Winner pays max virtual value among losers (original lines 87-91)
        # Create mask: ones everywhere except diagonal (to get "other agents")
        w_p = 1.0 - torch.eye(self.n_agents, device=bids.device)  # (agents, agents)

        # For each agent, compute max of other agents' virtual values
        # vv shape: (batch, agents), w_p: (agents, agents)
        vv_expanded = virtual_values.unsqueeze(1)  # (batch, 1, agents)
        masked_vv = vv_expanded * w_p  # (batch, agents, agents)
        p_spa_virtual = masked_vv.max(dim=2)[0]  # (batch, agents) - max over other agents

        # Decode payment: Inverse of encoding formula (original lines 94-100)
        # Formula: min_u max_l ((p_spa - w2) / exp(w1))
        p_expanded = p_spa_virtual.unsqueeze(1).unsqueeze(2)  # (batch, 1, 1, agents)
        w1_exp = self.w1.unsqueeze(0).expand(batch_size, -1, -1, -1)  # (batch, max_units, funcs, agents)
        w2_exp = self.w2.unsqueeze(0).expand(batch_size, -1, -1, -1)

        # Decode: (p_spa - w2) / exp(w1)
        p_decoded_interim = (p_expanded - w2_exp) / torch.exp(w1_exp)  # (batch, max_units, funcs, agents)
        p_max_funcs = p_decoded_interim.min(dim=2)[0]  # (batch, max_units, agents)
        payments = p_max_funcs.max(dim=1)[0]  # (batch, agents)

        # Revenue: expected payment across batch
        revenue = (allocations * payments).sum(dim=1).mean()

        return allocations, payments, revenue


def train_myerson_net(
    train_bids: np.ndarray,
    n_agents: Optional[int] = None,
    n_epochs: int = 50000,
    batch_size: int = 64,
    learning_rate: float = 0.001,
    n_linear_funcs: int = 10,
    n_max_units: int = 10,
    temperature: float = 1000.0,
    device: str = 'cpu',
    verbose: bool = True,
) -> MyersonNet:
    """Train MyersonNet on auction bid data.

    Args:
        train_bids: Training data of shape (n_auctions, n_agents) or (n_samples,)
            If 1D, will be reshaped assuming equal agents
        n_agents: Number of agents (inferred from data if None)
        n_epochs: Number of training iterations
        batch_size: Mini-batch size
        learning_rate: Adam optimizer learning rate
        n_linear_funcs: Number of linear functions in network
        n_max_units: Number of max units in network
        temperature: Softmax temperature for allocation
        device: 'cpu' or 'cuda'
        verbose: Print training progress

    Returns:
        Trained MyersonNet model
    """
    # Prepare data
    if train_bids.ndim == 1:
        # Assume data needs reshaping
        if n_agents is None:
            raise ValueError("n_agents must be specified for 1D bid data")
        n_samples = len(train_bids) // n_agents
        train_bids = train_bids[:n_samples * n_agents].reshape(n_samples, n_agents)
    else:
        n_agents = train_bids.shape[1]

    # Convert to torch tensor
    train_tensor = torch.FloatTensor(train_bids).to(device)

    # Initialize model
    model = MyersonNet(
        n_agents=n_agents,
        n_linear_funcs=n_linear_funcs,
        n_max_units=n_max_units,
        temperature=temperature,
    ).to(device)

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    n_samples = len(train_tensor)

    if verbose:
        print(f"Training MyersonNet: {n_agents} agents, {n_samples} samples")
        print(f"Epochs: {n_epochs}, Batch size: {batch_size}, LR: {learning_rate}")

    model.train()
    for epoch in range(n_epochs):
        # Sample random mini-batch
        indices = torch.randint(0, n_samples, (batch_size,))
        batch = train_tensor[indices]

        # Forward pass
        _, _, revenue = model(batch, training=True)

        # Loss: negative revenue (we want to maximize revenue)
        loss = -revenue

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Logging
        if verbose and (epoch + 1) % 10000 == 0:
            print(f"Epoch {epoch + 1}/{n_epochs}: Revenue = {revenue.item():.4f}")

    model.eval()
    return model


def compute_myerson_net_reserve_price(
    model: MyersonNet,
    lower_bound: float = 0.0,
    upper_bound: float = 10.0,
    device: str = 'cpu',
) -> float:
    """Compute optimal reserve price from trained MyersonNet.

    The reserve price is the bid where virtual value = 0.
    Bidders with bids below this have negative virtual values and reduce revenue.

    This implements Myerson's optimal reserve price using the learned
    virtual value transformation from the neural network.

    Args:
        model: Trained MyersonNet model
        lower_bound: Lower bound for search (default: 0.0)
        upper_bound: Upper bound for search (default: 10.0)
        device: 'cpu' or 'cuda'

    Returns:
        Optimal reserve price (bid where virtual value = 0)

    Example:
        >>> model = train_myerson_net(train_bids, ...)
        >>> reserve = compute_myerson_net_reserve_price(model)
        >>> print(f"Optimal reserve: {reserve:.4f}")
    """
    model.eval()

    # Binary search to find bid where virtual value = 0
    low, high = lower_bound, upper_bound
    tolerance = 1e-4
    max_iterations = 100

    for _ in range(max_iterations):
        mid = (low + high) / 2.0

        # Create test bid at midpoint for all agents
        # (we compute VV for one agent, others don't matter for this calculation)
        test_bid = torch.FloatTensor([[mid, mid, mid]]).to(device)

        with torch.no_grad():
            # Compute virtual values
            bids_expanded = test_bid.unsqueeze(1).unsqueeze(2)
            virtual_interim = bids_expanded * torch.exp(model.w1) + model.w2
            virtual_max = torch.max(virtual_interim, dim=2)[0]
            virtual_values = torch.min(virtual_max, dim=1)[0]

            # Get virtual value for first agent
            vv = virtual_values[0, 0].item()

        # Binary search: find where vv crosses zero
        if abs(vv) < tolerance:
            return float(mid)
        elif vv < 0:
            low = mid  # VV is negative, need higher bid
        else:
            high = mid  # VV is positive, can lower bid

    # Return midpoint if we hit max iterations
    return float((low + high) / 2.0)


def myerson_net_pricing(
    bids: np.ndarray,
    model: MyersonNet,
    use_reserve_price: bool = False,
    device: str = 'cpu',
) -> float:
    """Use trained MyersonNet to determine optimal price.

    Two modes:
    1. Expected payment mode (default): Computes expected payment from allocation/payment
    2. Reserve price mode: Computes Myerson optimal reserve price

    Args:
        bids: Bid data for current auction (any length)
        model: Trained MyersonNet model
        use_reserve_price: If True, return Myerson reserve price instead of expected payment
        device: 'cpu' or 'cuda'

    Returns:
        Optimal price (either expected payment or reserve price)

    Note:
        If bids length > model.n_agents, reshapes into multiple samples and averages prices.
        If bids length < model.n_agents, pads with min bid value.
        If bids length % model.n_agents != 0, truncates excess bids.

    Example:
        >>> # Mode 1: Expected payment (second-price like)
        >>> price = myerson_net_pricing(bids, model)
        >>>
        >>> # Mode 2: Myerson reserve price
        >>> reserve = myerson_net_pricing(bids, model, use_reserve_price=True)
    """
    if use_reserve_price:
        # Return Myerson optimal reserve price
        return compute_myerson_net_reserve_price(model, device=device)

    # Original behavior: expected payment
    model.eval()

    # Handle variable-length bids
    n_bids = len(bids)
    n_agents = model.n_agents

    if n_bids < n_agents:
        # Pad with minimum bid value if too few bids
        padding = np.full(n_agents - n_bids, bids.min())
        bids = np.concatenate([bids, padding])
        n_bids = n_agents

    if n_bids % n_agents != 0:
        # Truncate excess bids to make divisible by n_agents
        n_bids = (n_bids // n_agents) * n_agents
        bids = bids[:n_bids]

    # Reshape into multiple samples of n_agents each
    n_samples = n_bids // n_agents
    bids_reshaped = bids.reshape(n_samples, n_agents)

    # Convert to tensor (batch dimension already from reshape)
    bids_tensor = torch.FloatTensor(bids_reshaped).to(device)

    with torch.no_grad():
        allocations, payments, _ = model(bids_tensor, training=False)

    # Compute price for each sample, then average
    prices = []
    for i in range(n_samples):
        alloc_sum = allocations[i].sum().item()
        if alloc_sum > 0:
            sample_price = (payments[i] * allocations[i]).sum().item() / alloc_sum
            prices.append(sample_price)
        else:
            prices.append(0.0)

    # Return average price across all samples
    return float(np.mean(prices) if prices else 0.0)


# Model I/O utilities
def save_myerson_net(model: MyersonNet, filepath: Path) -> None:
    """Save trained MyersonNet model.

    Args:
        model: Trained model
        filepath: Path to save file (.pt)
    """
    filepath.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        'state_dict': model.state_dict(),
        'n_agents': model.n_agents,
        'n_linear_funcs': model.n_linear_funcs,
        'n_max_units': model.n_max_units,
        'temperature': model.temperature,
    }, filepath)
    print(f"✅ Saved MyersonNet model ({filepath.stat().st_size:,} bytes)")


def load_myerson_net(filepath: Path, device: str = 'cpu') -> Optional[MyersonNet]:
    """Load trained MyersonNet model.

    Args:
        filepath: Path to saved model (.pt)
        device: 'cpu' or 'cuda'

    Returns:
        Loaded model or None if file doesn't exist
    """
    if not filepath.exists():
        return None

    checkpoint = torch.load(filepath, map_location=device)
    model = MyersonNet(
        n_agents=checkpoint['n_agents'],
        n_linear_funcs=checkpoint['n_linear_funcs'],
        n_max_units=checkpoint['n_max_units'],
        temperature=checkpoint['temperature'],
    ).to(device)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    print(f"📂 Loaded MyersonNet model: {filepath.name}")
    return model
