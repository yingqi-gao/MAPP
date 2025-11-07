"""Load experimental regret results from workspace."""

import pickle
import warnings
from pathlib import Path
from typing import Any, Dict, List, Union


def _get_display_name(method_slug: str) -> str:
    """Convert method slug from filename to display name.

    Examples:
        'ecdf' → 'ecdf'
        'rde_200x20' → 'RDE (200x20)'
        'rde_20x200' → 'RDE (20x200)'
        'myersonnet_2a' → 'MyersonNet'
        'myersonnet_100a' → 'MyersonNet'
    """
    # Handle RDE variants: rde_NxN → RDE (NxN)
    if method_slug.startswith('rde_'):
        config = method_slug[4:]  # Remove 'rde_' prefix
        return f'RDE ({config})'

    # Handle MyersonNet variants: myersonnet_*a → MyersonNet
    if method_slug.startswith('myersonnet_'):
        return 'MyersonNet'

    # Return as-is for other methods
    return method_slug


def _get_method_slug(display_name: str) -> str:
    """Convert display name to method slug for filtering.

    Examples:
        'ecdf' → 'ecdf'
        'RDE (200x20)' → 'rde_200x20'
        'RDE (20x200)' → 'rde_20x200'
        'MyersonNet' → None (will match any myersonnet_* slug)
    """
    # Handle RDE variants: RDE (NxN) → rde_NxN
    if display_name.startswith('RDE (') and display_name.endswith(')'):
        config = display_name[5:-1]  # Extract config from "RDE (config)"
        return f'rde_{config}'

    # Handle MyersonNet: return None to signal we need special matching
    if display_name == 'MyersonNet':
        return None  # Special case: will match any myersonnet_* slug

    # Return as-is for other methods
    return display_name


def load_results(
    project_root: Path,
    dist_name: Union[str, List[str]],
    bids_per_auction: Union[int, List[int], None] = None,
    k_folds: Union[int, List[int], None] = None,
    methods: Union[str, List[str], None] = None,
) -> List[Dict[str, Any]]:
    """Load experimental results from workspace.

    Args:
        project_root: Project root directory (workspace = project_root / "workspace")
        dist_name: Distribution name(s) to load
        bids_per_auction: Bid count(s) to filter (None = all)
        k_folds: K-fold value(s) to filter (None = all)
        methods: Method(s) to filter (None = all)

    Returns:
        List of result dicts with keys: method, n_runs, n_auctions,
        bids_per_auction, regrets, dist_name, k_folds, file_path
    """
    workspace_path = project_root / "workspace"

    # Normalize inputs
    dists = [dist_name] if isinstance(dist_name, str) else dist_name
    bids = [bids_per_auction] if isinstance(bids_per_auction, int) else bids_per_auction
    ks = [k_folds] if isinstance(k_folds, int) else k_folds
    meths = [methods] if isinstance(methods, str) else methods

    # Convert display names to slugs for filtering
    method_slugs = None
    if meths:
        method_slugs = []
        has_myersonnet = False
        for m in meths:
            slug = _get_method_slug(m)
            if slug is None:  # MyersonNet case
                has_myersonnet = True
            else:
                method_slugs.append(slug)
        # Store flag for MyersonNet special handling
        method_slugs_set = set(method_slugs) if method_slugs else set()
    else:
        has_myersonnet = False
        method_slugs_set = None

    results = []
    for dist in dists:
        regrets_dir = workspace_path / dist / "regrets"
        if not regrets_dir.exists():
            warnings.warn(f"Not found: {regrets_dir}")
            continue

        for pkl_file in regrets_dir.glob("*.pkl"):
            # Parse filename: {dist}_b{bids}_k{k}_{method}.pkl
            parts = pkl_file.stem.split('_')
            try:
                file_bids, file_k, method_parts = None, None, []
                for part in parts[1:]:  # Skip dist name
                    if part.startswith('b') and part[1:].isdigit():
                        file_bids = int(part[1:])
                    elif part.startswith('k') and part[1:].isdigit():
                        file_k = int(part[1:])
                    else:
                        method_parts.append(part)

                file_method = '_'.join(method_parts)

                # Apply filters
                if bids and file_bids not in bids:
                    continue
                if ks and file_k not in ks:
                    continue
                if meths:
                    # Check if file_method matches any of the requested methods
                    is_myersonnet = file_method.startswith('myersonnet_')
                    if is_myersonnet:
                        # MyersonNet file: only include if MyersonNet was requested
                        if not has_myersonnet:
                            continue
                    else:
                        # Non-MyersonNet file: check against method slugs
                        if file_method not in method_slugs_set:
                            continue

                # Load file
                with open(pkl_file, 'rb') as f:
                    result = pickle.load(f)

                # Override method name with display name from filename
                result['method'] = _get_display_name(file_method)

                result['dist_name'] = dist
                result['k_folds'] = file_k
                result['file_path'] = pkl_file
                results.append(result)

            except (ValueError, IndexError) as e:
                warnings.warn(f"Could not parse {pkl_file.name}: {e}")

    if not results:
        warnings.warn(f"No results found for dist={dists}, bids={bids}, k={ks}, methods={meths}")

    return results
