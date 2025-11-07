"""Generic filename generation utilities."""

from pathlib import Path
from typing import Literal

# Get project root (2 levels up from this file: mapp/utils/filename.py -> mapp/ -> project_root/)
_PROJECT_ROOT = Path(__file__).parent.parent.parent


def _format_value(value) -> str:
    """Format value for filename: integers without decimals, floats with decimals.

    Args:
        value: Any value to format

    Returns:
        Formatted string representation

    Examples:
        >>> _format_value(1.0)
        '1'
        >>> _format_value(1.5)
        '1.5'
        >>> _format_value(10.0)
        '10'
        >>> _format_value("test")
        'test'
    """
    if isinstance(value, float) and value == int(value):
        return str(int(value))
    return str(value)


def generate_filename(
    dist_name: str,
    file_type: Literal["data", "regrets", "rde_models"],
    **kwargs,
) -> Path:
    """Generate standardized filename path organized by distribution.

    Args:
        dist_name: Distribution name ('truncnorm', 'truncexpon', 'beta', 'truncpareto', 'real')
        file_type: "data", "regrets", or "rde_models" (determines subfolder)
        **kwargs: Any key-value pairs to include in filename
            Common parameters (abbreviated for conciseness):
            - r: number of runs
            - a: number of auctions per run
            - b: number of bids per auction
            - s: random seed
            - N: number of training auctions (RDE models)
            - n: number of training bids (RDE models)
            - l: lower bound (RDE models)
            - u: upper bound (RDE models)
            - purpose: data purpose ('train' or 'test'), appended at end without prefix
            - method: pricing method name, appended at end without prefix
            - subfolder: optional additional subfolder (e.g., for organizing RDE models by training data)

    Returns:
        Full path to file

    Examples:
        >>> # Test data with multiple runs
        >>> generate_filename("truncnorm", file_type="data", r=100, a=50, b=50, s=100, purpose="test")
        Path('workspace/truncnorm/data/truncnorm_a50_b50_r100_s100_test.json')

        >>> # Training data (no r parameter, has purpose='train')
        >>> generate_filename("truncnorm", file_type="data", a=200, b=200, s=42, purpose="train")
        Path('workspace/truncnorm/data/truncnorm_a200_b200_s42_train.json')

        >>> # RDE model organized by training data source
        >>> generate_filename("truncnorm", file_type="rde_models", subfolder="a200_b200_s42", N=50, n=10, l=1.0, u=10.0)
        Path('workspace/truncnorm/rde_models/a200_b200_s42/truncnorm_N50_l1_n10_u10.json')

        >>> # Regret results file with method
        >>> generate_filename("truncnorm", file_type="regrets", b=50, k=10, method="ecdf")
        Path('workspace/truncnorm/regrets/truncnorm_b50_k10_ecdf.json')
    """
    # Extract special parameters that affect path structure or naming (not prefixed)
    purpose = kwargs.pop('purpose', None)
    method = kwargs.pop('method', None)
    subfolder = kwargs.pop('subfolder', None)

    # Build filename parts from remaining kwargs
    parts = [dist_name]
    for key, value in sorted(kwargs.items()):
        if value is not None:
            parts.append(f"{key}{_format_value(value)}")

    # Append method and purpose at the end if provided (without prefix)
    if method is not None:
        parts.append(method)
    if purpose is not None:
        parts.append(purpose)

    filename = "_".join(parts) + ".json"

    # Build path with optional subfolder (absolute path from project root)
    base_path = _PROJECT_ROOT / "workspace" / dist_name / file_type
    if subfolder is not None:
        base_path = base_path / subfolder

    return base_path / filename
