"""Distribution parameter generation for auction simulation.

Parameter ranges are carefully chosen to create distributions that are:
- Non-trivial to estimate (not too simple/degenerate)
- Computationally efficient (don't require large training sets or long test times)
- Realistic for auction pricing experiments
"""

import numpy as np

from mapp.utils.constants import VALUE_LOWER_BOUND, VALUE_UPPER_BOUND

# Truncated Normal: mean centered around middle, moderate spread
TRUNCNORM_MU_RANGE = (0.45, 0.65)      # Mean as fraction of span
TRUNCNORM_SIGMA_RANGE = (0.10, 0.30)   # Standard deviation as fraction of span

# Truncated Exponential: moderate decay rates
TRUNCEXPON_SCALE_RANGE = (0.20, 0.60)  # Scale as fraction of span

# Beta: skewed but not extreme, moderate variance
BETA_A_RANGE = (5.0, 10.0)             # Shape parameter a
BETA_MEAN_FRAC_RANGE = (0.40, 0.80)    # Target mean as fraction of span

# Truncated Pareto: light-tailed (alpha > 1), moderate scale
TRUNCPARETO_ALPHA_RANGE = (1.15, 1.30)  # Tail index (higher = lighter tail)
TRUNCPARETO_SCALE_RANGE = (0.30, 0.80)  # Scale as fraction of span


def _generate_truncnorm_params(
    span: float, lower: float, rng: np.random.Generator
) -> dict[str, float]:
    """Generate truncated normal parameters (a, b, loc, scale)."""
    mu = rng.uniform(lower + TRUNCNORM_MU_RANGE[0] * span, lower + TRUNCNORM_MU_RANGE[1] * span)
    sigma = rng.uniform(TRUNCNORM_SIGMA_RANGE[0] * span, TRUNCNORM_SIGMA_RANGE[1] * span)
    return {
        "a": (lower - mu) / sigma,
        "b": (lower + span - mu) / sigma,
        "loc": mu,
        "scale": sigma,
    }


def _generate_truncexpon_params(
    span: float, lower: float, rng: np.random.Generator
) -> dict[str, float]:
    """Generate truncated exponential parameters (b, loc, scale)."""
    scale = rng.uniform(*TRUNCEXPON_SCALE_RANGE) * span
    return {"b": span / scale, "loc": lower, "scale": scale}


def _generate_beta_params(
    span: float, lower: float, rng: np.random.Generator
) -> dict[str, float]:
    """Generate beta parameters (a, b, loc, scale)."""
    a = rng.uniform(*BETA_A_RANGE)
    mean_frac = rng.uniform(*BETA_MEAN_FRAC_RANGE)
    b = a * (1.0 / mean_frac - 1.0)  # From: mean = a/(a+b) => b = a(1/mean - 1)
    return {"a": a, "b": b, "loc": lower, "scale": span}


def _generate_truncpareto_params(
    span: float, lower: float, rng: np.random.Generator
) -> dict[str, float]:
    """Generate truncated Pareto parameters (b, c, loc, scale)."""
    alpha = rng.uniform(*TRUNCPARETO_ALPHA_RANGE)
    scale = rng.uniform(TRUNCPARETO_SCALE_RANGE[0] * span, TRUNCPARETO_SCALE_RANGE[1] * span)
    c = 1.0 + (span / scale)  # Upper bound ratio for truncation
    return {"b": alpha, "c": c, "loc": lower - scale, "scale": scale}


# Distribution parameter generators
PARAM_GENERATORS = {
    "truncnorm": _generate_truncnorm_params,
    "truncexpon": _generate_truncexpon_params,
    "beta": _generate_beta_params,
    "truncpareto": _generate_truncpareto_params,
}


def _generate_dist_params(
    dist_name: str,
    *,
    rng: np.random.Generator,
) -> dict[str, float]:
    """Generate random parameters for a distribution family.

    Args:
        dist_name: Distribution family name
        rng: Random number generator

    Returns:
        Parameter dictionary for scipy distribution
    """
    if dist_name not in PARAM_GENERATORS:
        valid = list(PARAM_GENERATORS.keys())
        raise ValueError(f"Unknown distribution: '{dist_name}'. Valid: {valid}")

    span = VALUE_UPPER_BOUND - VALUE_LOWER_BOUND
    generator = PARAM_GENERATORS[dist_name]
    return generator(span, VALUE_LOWER_BOUND, rng)
