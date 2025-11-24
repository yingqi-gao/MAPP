"""Visualization utilities for MAPP experimental results."""

import warnings
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterator, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from mapp.utils.regrets_loader import load_results


# Configuration
DEFAULT_DPI = 600

# Styling - Publication-quality configuration
_METHOD_CONFIG = {
    "ecdf": {"color": "#2C3E50", "linestyle": "-", "marker": "o", "order": 0},  # Dark blue-gray
    "kde": {"color": "#3498DB", "linestyle": "--", "marker": "s", "order": 1},  # Bright blue
    "myerson": {"color": "#27AE60", "linestyle": "-.", "marker": "D", "order": 2},  # Green
    "RDE (20x200)": {"color": "#F39C12", "linestyle": "-", "marker": "^", "order": 3},  # Orange - few dense auctions
    "RDE (200x20)": {"color": "#E67E22", "linestyle": "--", "marker": "v", "order": 4},  # Dark orange - many sparse auctions
    "RDE (200x200)": {"color": "#C0392B", "linestyle": "-.", "marker": "p", "order": 5},  # Dark red - full data (best)
    "MyersonNet": {"color": "#9B59B6", "linestyle": "-", "marker": "h", "order": 6},  # Purple - neural network method
}

_DIST_NAMES = {
    "truncnorm": "Truncated Normal", 
    "truncexpon": "Truncated Exponential",
    "beta": "Beta", 
    "truncpareto": "Truncated Pareto",
}


def _get_style(name: str, palette: Iterator[Any]) -> Tuple[str, str]:
    """Get color and linestyle for method."""
    config = _METHOD_CONFIG.get(name)
    if config:
        return config["color"], config["linestyle"]
    return next(palette), "-"


def _get_order(name: str) -> Tuple[int, str]:
    """Get sort order for method."""
    config = _METHOD_CONFIG.get(name)
    if config:
        return config["order"], name
    return len(_METHOD_CONFIG), name


def _get_marker(name: str) -> str:
    """Get marker style for method."""
    config = _METHOD_CONFIG.get(name)
    if config:
        return config["marker"]
    return "o"


# Plotting
def plot_kfold_sensitivity(
    project_root: Path,
    dist_names: Union[str, List[str]],
    bids_per_auction: Union[int, List[int]],
    k_folds: Optional[List[int]] = None,
    methods: Optional[List[str]] = None,
    figsize: Tuple[float, float] = (10, 6),
    dpi: int = 300,
    save_dir: Optional[Union[str, Path]] = None,
    show: bool = False,
    suffix: Optional[str] = None,
) -> None:
    """Plot k-fold sensitivity analysis for methods across different k values.

    Creates one figure per distribution with 4 subplots (2x2 grid), each showing
    k-fold sensitivity for a different bid count, with a shared legend on the right.

    Args:
        project_root: Project root directory (workspace = project_root / "workspace")
        dist_names: Distribution(s) to plot
        bids_per_auction: Bid count(s) to plot (creates one subplot per value, max 4)
        k_folds: K-fold values to compare (None = all available)
        methods: Methods to include (None = all available)
        figsize: Figure size (width, height) - ignored, uses (16, 10)
        dpi: Resolution
        save_dir: Directory to save plots (None = don't save)
                 Relative paths are relative to project_root
        show: Display plots
        suffix: Optional suffix for filename (e.g., '1', '2', '3')

    Examples:
        >>> from pathlib import Path
        >>> project_root = Path.cwd().parent
        >>>
        >>> # Single distribution with 4 bid counts in one plot
        >>> plot_kfold_sensitivity(project_root, 'truncnorm', [10, 50, 100, 200])
        >>>
        >>> # All distributions (creates 4 plots, one per distribution)
        >>> plot_kfold_sensitivity(
        ...     project_root,
        ...     ['truncnorm', 'truncexpon', 'beta', 'truncpareto'],
        ...     [10, 50, 100, 200],
        ...     save_dir='workspace/plots/sensitivity'
        ... )
    """
    # Normalize inputs
    dists = [dist_names] if isinstance(dist_names, str) else dist_names
    bids_list = [bids_per_auction] if isinstance(bids_per_auction, int) else bids_per_auction

    # Set up save directory if provided
    save_path: Optional[Path] = None
    if save_dir is not None:
        save_path = Path(save_dir)
        # If relative path, make it relative to project_root
        if not save_path.is_absolute():
            save_path = project_root / save_path
        save_path.mkdir(parents=True, exist_ok=True)

    for dist_name in dists:
        # Create figure with 2x2 subplots
        sns.set_style("whitegrid")
        fig, axes = plt.subplots(2, 2, figsize=(16, 10), dpi=dpi)
        axes_flat = axes.flatten()

        dist_display = _DIST_NAMES.get(dist_name.lower(), dist_name)
        fig.suptitle(
            f'{dist_display}: K-Fold Sensitivity Analysis',
            fontsize=28, fontweight='bold', y=0.995
        )

        # Track all methods for shared legend
        all_method_handles = {}
        all_method_labels = {}

        # Plot each bid count in a subplot
        for idx, n_bids in enumerate(bids_list[:4]):
            ax = axes_flat[idx]

            # Load results for this combination
            # NOTE: load_results() automatically filters MyersonNet files to only load
            # those with agent count matching group_size (bids/k). This ensures each
            # (bids, k) combination loads only the correct MyersonNet model.
            results = load_results(
                project_root=project_root,
                dist_name=dist_name,
                bids_per_auction=n_bids,
                k_folds=k_folds,
                methods=methods,
            )

            if not results:
                warnings.warn(f"No results for {dist_name}, n={n_bids}")
                ax.text(0.5, 0.5, 'No Data', ha='center', va='center', fontsize=16)
                ax.set_title(f'{n_bids} Bids per Auction', fontsize=14, fontweight='bold')
                continue

            # Extract k-fold data by method
            # Use dict to deduplicate: (method, k) → regrets
            method_k_regrets = defaultdict(list)

            for result in results:
                method = result['method']
                k = result.get('k_folds') or result.get('k', 0)
                regrets = np.array(result['regrets'])

                # Collect all regrets for this (method, k) combination
                method_k_regrets[(method, k)].append(regrets)

            # Aggregate: if multiple results for same (method, k), average the regrets
            method_data = defaultdict(lambda: {'k': [], 'mean': [], 'std': []})

            for (method, k), regrets_list in method_k_regrets.items():
                # If multiple results, concatenate all regrets and recompute stats
                all_regrets = np.concatenate(regrets_list) if len(regrets_list) > 1 else regrets_list[0]

                mean = np.mean(all_regrets)
                std = np.std(all_regrets)

                method_data[method]['k'].append(k)
                method_data[method]['mean'].append(mean)
                method_data[method]['std'].append(std)

            palette_iter = iter(sns.color_palette())

            for method in sorted(method_data.keys(), key=_get_order):
                data = method_data[method]
                if not data['k']:
                    continue

                # Sort by k value
                sorted_indices = sorted(range(len(data['k'])), key=lambda i: data['k'][i])
                k_vals = np.array([data['k'][i] for i in sorted_indices])
                means = np.array([data['mean'][i] for i in sorted_indices])
                stds = np.array([data['std'][i] for i in sorted_indices])

                # Get style
                color, linestyle = _get_style(method, palette_iter)
                marker = _get_marker(method)

                # Check if MyersonNet has invalid scenarios (group_size < 2)
                # For MyersonNet, skip error bars where group_size < 2
                is_myersonnet = method.startswith("MyersonNet")
                if is_myersonnet:
                    # Filter out k values where group_size < 2
                    valid_indices = [i for i, k in enumerate(k_vals) if n_bids // k >= 2]

                    if valid_indices:
                        valid_k = k_vals[valid_indices]
                        valid_means = means[valid_indices]
                        valid_stds = stds[valid_indices]

                        # Plot line for valid points only
                        ax.plot(
                            valid_k, valid_means,
                            color=color, linestyle=linestyle, linewidth=2.5,
                            zorder=2, alpha=0.7
                        )

                        # Plot error bars with markers for valid points
                        err = ax.errorbar(
                            valid_k, valid_means, yerr=valid_stds,
                            label=method, color=color, linestyle='none',
                            marker=marker, markersize=10,
                            markeredgewidth=2, markeredgecolor='white',
                            capsize=5, capthick=2, elinewidth=2,
                            zorder=3, alpha=0.9
                        )

                        # Store for shared legend (only once)
                        if method not in all_method_handles:
                            all_method_handles[method] = err
                            all_method_labels[method] = method
                else:
                    # For non-MyersonNet methods, plot normally
                    # Plot line connecting means
                    ax.plot(
                        k_vals, means,
                        color=color, linestyle=linestyle, linewidth=2.5,
                        zorder=2, alpha=0.7
                    )

                    # Plot vertical error bars with markers
                    err = ax.errorbar(
                        k_vals, means, yerr=stds,
                        label=method, color=color, linestyle='none',
                        marker=marker, markersize=10,
                        markeredgewidth=2, markeredgecolor='white',
                        capsize=5, capthick=2, elinewidth=2,
                        zorder=3, alpha=0.9
                    )

                    # Store for shared legend (only once)
                    if method not in all_method_handles:
                        all_method_handles[method] = err
                        all_method_labels[method] = method

            # Formatting
            ax.set_xlabel('K-Fold Value', fontsize=22)
            ax.set_ylabel('Mean Regret', fontsize=22)
            ax.set_title(f'{n_bids} Bids per Auction', fontsize=24, fontweight='bold', pad=10)

            # Better grid
            ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.8, zorder=0)
            ax.set_axisbelow(True)

            # Remove top and right spines
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_linewidth(1.5)
            ax.spines['bottom'].set_linewidth(1.5)

            # Set x-ticks to actual k values
            if method_data:
                all_k = sorted(set(k for data in method_data.values() for k in data['k']))
                ax.set_xticks(all_k)
                ax.tick_params(axis='both', labelsize=20, width=1.5, length=6)

        # Add shared legend on the right side
        if all_method_handles:
            # Sort handles and labels by method order
            sorted_methods = sorted(all_method_handles.keys(), key=_get_order)
            handles = [all_method_handles[m] for m in sorted_methods]
            labels = [all_method_labels[m] for m in sorted_methods]

            fig.legend(
                handles, labels,
                fontsize=22, loc='center left', bbox_to_anchor=(0.92, 0.5),
                frameon=True, fancybox=True, shadow=True, framealpha=0.95,
                edgecolor='gray', ncol=1
            )

        plt.tight_layout(rect=[0, 0, 0.90, 0.99])  # Leave space for legend on right

        # Save
        if save_path:
            if suffix:
                filename = f'kfold_sensitivity_{dist_name}_{suffix}.png'
            else:
                filename = f'kfold_sensitivity_{dist_name}.png'
            plt.savefig(save_path / filename, dpi=dpi, bbox_inches='tight')
            bids_str = ', '.join(str(b) for b in bids_list[:4])
            print(f"✅ Saved: {filename} ({len(bids_list[:4])} subplots: {bids_str} bids)")

        # Show or close
        if show:
            plt.show()
        else:
            plt.close(fig)


def plot_regret_histograms(
    project_root: Path,
    dist_names: Union[str, List[str]],
    bids_per_auction: List[int],
    k_fold: int = 2,
    methods: Optional[List[str]] = None,
    bins: int = 30,
    alpha: float = 0.6,
    figsize: Tuple[float, float] = (16, 10),
    dpi: int = 300,
    save_dir: Optional[Union[str, Path]] = None,
    show: bool = False,
    suffix: Optional[str] = None,
) -> None:
    """Plot histogram comparisons across methods for different bid counts.

    Creates one figure per distribution with 4 subplots (2x2 grid), each showing
    overlapping histograms of regret distributions for a different bid count.

    Args:
        project_root: Project root directory (workspace = project_root / "workspace")
        dist_names: Distribution(s) to plot
        bids_per_auction: List of bid counts (creates one subplot per value)
        k_fold: K-fold value to use (default: 2)
        methods: Methods to include (None = all available)
        bins: Number of histogram bins
        alpha: Transparency for overlapping histograms
        figsize: Figure size (width, height)
        dpi: Resolution
        save_dir: Directory to save plots (None = don't save)
                 Relative paths are relative to project_root
        show: Display plots
        suffix: Optional suffix for filename (e.g., '1', '2', '3')

    Examples:
        >>> from pathlib import Path
        >>> project_root = Path.cwd().parent
        >>>
        >>> # Single distribution with 4 bid counts
        >>> plot_regret_histograms(project_root, 'truncnorm', [10, 50, 100, 200], k_fold=2)
        >>>
        >>> # All distributions
        >>> plot_regret_histograms(
        ...     project_root,
        ...     ['truncnorm', 'truncexpon', 'beta', 'truncpareto'],
        ...     [10, 50, 100, 200],
        ...     save_dir='workspace/plots/histograms'
        ... )
    """
    # Normalize inputs
    dists = [dist_names] if isinstance(dist_names, str) else dist_names

    # Set up save directory if provided
    save_path: Optional[Path] = None
    if save_dir is not None:
        save_path = Path(save_dir)
        if not save_path.is_absolute():
            save_path = project_root / save_path
        save_path.mkdir(parents=True, exist_ok=True)

    for dist_name in dists:
        # Create figure with 2x2 subplots
        sns.set_style("whitegrid")
        fig, axes = plt.subplots(2, 2, figsize=figsize, dpi=dpi)
        axes_flat = axes.flatten()

        dist_display = _DIST_NAMES.get(dist_name.lower(), dist_name)
        fig.suptitle(
            f'{dist_display}: Regret Distribution Comparison',
            fontsize=28, fontweight='bold', y=0.995
        )

        # Track all methods for shared legend
        all_method_handles = {}
        all_method_labels = {}

        # Plot each bid count in a subplot
        for idx, n_bids in enumerate(bids_per_auction[:4]):  # Limit to 4 subplots
            ax = axes_flat[idx]

            # Load results for this combination
            # NOTE: load_results() automatically filters MyersonNet files to only load
            # those with agent count matching group_size (bids/k). This ensures each
            # (bids, k) combination loads only the correct MyersonNet model.
            results = load_results(
                project_root=project_root,
                dist_name=dist_name,
                bids_per_auction=n_bids,
                k_folds=k_fold,
                methods=methods,
            )

            if not results:
                warnings.warn(f"No results for {dist_name}, n={n_bids}, k={k_fold}")
                ax.text(0.5, 0.5, 'No Data', ha='center', va='center', fontsize=16)
                ax.set_title(f'{n_bids} Bids', fontsize=14, fontweight='bold')
                continue

            # Deduplicate: if multiple results for same method, merge regrets
            method_regrets = defaultdict(list)
            for result in results:
                method = result['method']
                regrets = np.array(result['regrets'])
                method_regrets[method].append(regrets)

            # Plot histograms for each method
            palette_iter = iter(sns.color_palette())

            for method in sorted(method_regrets.keys(), key=_get_order):
                # Merge all regrets for this method (handles MyersonNet variants)
                regrets_list = method_regrets[method]
                all_regrets = np.concatenate(regrets_list) if len(regrets_list) > 1 else regrets_list[0]

                color, _ = _get_style(method, palette_iter)

                # Plot histogram
                _, _, patches = ax.hist(
                    all_regrets, bins=bins, alpha=alpha, color=color,
                    label=f"{method}",
                    edgecolor='white', linewidth=0.5
                )

                # Store for shared legend (only once)
                if method not in all_method_handles:
                    all_method_handles[method] = patches[0]
                    all_method_labels[method] = method

            # Formatting
            ax.set_xlabel('Regret', fontsize=22)
            ax.set_ylabel('Frequency', fontsize=22)
            ax.set_title(f'{n_bids} Bids per Auction', fontsize=24, fontweight='bold', pad=10)

            # Grid
            ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.8, zorder=0, axis='y')
            ax.set_axisbelow(True)

            # Remove top and right spines
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_linewidth(1.5)
            ax.spines['bottom'].set_linewidth(1.5)

            ax.tick_params(axis='both', labelsize=14, width=1.5, length=6)

        # Add shared legend on the right side
        if all_method_handles:
            # Sort handles and labels by method order
            sorted_methods = sorted(all_method_handles.keys(), key=_get_order)
            handles = [all_method_handles[m] for m in sorted_methods]
            labels = [all_method_labels[m] for m in sorted_methods]

            fig.legend(
                handles, labels,
                fontsize=22, loc='center left', bbox_to_anchor=(0.92, 0.5),
                frameon=True, fancybox=True, shadow=True, framealpha=0.95,
                edgecolor='gray', ncol=1
            )

        plt.tight_layout(rect=[0, 0, 0.90, 0.99])  # Leave space for legend on right

        # Save
        if save_path:
            if suffix:
                filename = f'histogram_{dist_name}_k{k_fold}_{suffix}.png'
            else:
                filename = f'histogram_{dist_name}_k{k_fold}.png'
            plt.savefig(save_path / filename, dpi=dpi, bbox_inches='tight')
            bids_str = ', '.join(str(b) for b in bids_per_auction[:4])
            print(f"✅ Saved: {filename} ({len(bids_per_auction[:4])} subplots: {bids_str} bids)")

        # Show or close
        if show:
            plt.show()
        else:
            plt.close(fig)
