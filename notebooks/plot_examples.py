"""Example usage of the new plot.py utilities.

This script demonstrates how to use the rewritten plotting utilities
to load results from the workspace and create publication-quality figures.

Run this script from the project root:
    python notebooks/plot_examples.py
"""

from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from mapp.utils.plot import (
    load_results,
    group_results_by_bids,
    plot_regret_densities,
    plot_regret_boxplots,
    create_summary_table,
    compare_methods_across_dists,
    compute_statistical_significance,
)


def example_1_basic_density_plot():
    """Example 1: Load results and create basic density plot."""
    print("\n" + "="*80)
    print("Example 1: Basic Density Plot")
    print("="*80)

    # Load truncnorm results with k=2 folds
    results = load_results(
        dist_name='truncnorm',
        bids_per_auction=[10, 50, 100],
        k_folds=2,
        methods=['ecdf', 'kde', 'rde_200x200']
    )

    print(f"Loaded {len(results)} result files")

    # Group by bid count
    plot_data = group_results_by_bids(results)

    # Create density plot
    plot_regret_densities(
        plot_data,
        save_path='figures/example1_truncnorm_k2.png',
        show=False,
        title='Truncated Normal Distribution (k=2)',
        wlegend=True
    )

    print("✓ Plot saved to: figures/example1_truncnorm_k2.png")


def example_2_boxplot():
    """Example 2: Create box plot comparison."""
    print("\n" + "="*80)
    print("Example 2: Box Plot Comparison")
    print("="*80)

    results = load_results(
        dist_name='beta',
        bids_per_auction=[10, 50, 100],
        k_folds=2
    )

    print(f"Loaded {len(results)} result files")

    plot_data = group_results_by_bids(results)

    plot_regret_boxplots(
        plot_data,
        save_path='figures/example2_beta_boxplot.png',
        show=False,
        title='Beta Distribution - Method Comparison'
    )

    print("✓ Plot saved to: figures/example2_beta_boxplot.png")


def example_3_summary_table():
    """Example 3: Create summary statistics table."""
    print("\n" + "="*80)
    print("Example 3: Summary Statistics Table")
    print("="*80)

    results = load_results(
        dist_name='truncnorm',
        bids_per_auction=[10, 50, 100],
        k_folds=2,
        methods=['ecdf', 'kde', 'rde_200x200']
    )

    print(f"Loaded {len(results)} result files\n")

    # Create summary table
    table = create_summary_table(
        results,
        metrics=['mean', 'std', 'median', 'min', 'max'],
        output_format='dataframe'
    )

    print(table)

    # Export to LaTeX
    latex_table = create_summary_table(results, output_format='latex')
    with open('figures/example3_summary.tex', 'w') as f:
        f.write(latex_table)

    print("\n✓ LaTeX table saved to: figures/example3_summary.tex")

    # Export to Markdown
    md_table = create_summary_table(results, output_format='markdown')
    with open('figures/example3_summary.md', 'w') as f:
        f.write(md_table)

    print("✓ Markdown table saved to: figures/example3_summary.md")


def example_4_cross_distribution():
    """Example 4: Compare methods across distributions."""
    print("\n" + "="*80)
    print("Example 4: Cross-Distribution Comparison")
    print("="*80)

    # This creates a grid of plots comparing all distributions
    compare_methods_across_dists(
        dist_names=['truncnorm', 'truncexpon', 'beta', 'truncpareto'],
        bids_per_auction=[10, 50, 100],
        k_folds=2,
        methods=['ecdf', 'kde', 'rde_200x200'],
        save_path='figures/example4_cross_dist.png',
        show=False
    )

    print("✓ Plot saved to: figures/example4_cross_dist.png")


def example_5_statistical_testing():
    """Example 5: Statistical significance testing."""
    print("\n" + "="*80)
    print("Example 5: Statistical Significance Testing")
    print("="*80)

    results = load_results(
        dist_name='truncnorm',
        bids_per_auction=50,
        k_folds=2,
        methods=['ecdf', 'kde', 'rde_200x200']
    )

    # Extract regrets for each method
    ecdf_regrets = [r['regrets'] for r in results if r['method'] == 'ecdf'][0]
    kde_regrets = [r['regrets'] for r in results if r['method'] == 'kde'][0]
    rde_regrets = [r['regrets'] for r in results if r['method'] == 'rde_200x200'][0]

    # Compare eCDF vs KDE
    test1 = compute_statistical_significance(ecdf_regrets, kde_regrets)
    print(f"\neCDF vs KDE:")
    print(f"  Mean eCDF: {test1['mean1']:.4f}")
    print(f"  Mean KDE:  {test1['mean2']:.4f}")
    print(f"  p-value:   {test1['pvalue']:.4e}")
    print(f"  Significant: {test1['significant']}")
    print(f"  Effect size: {test1['effect_size']:.4f}")

    # Compare eCDF vs RDE
    test2 = compute_statistical_significance(ecdf_regrets, rde_regrets)
    print(f"\neCDF vs RDE (200×200):")
    print(f"  Mean eCDF: {test2['mean1']:.4f}")
    print(f"  Mean RDE:  {test2['mean2']:.4f}")
    print(f"  p-value:   {test2['pvalue']:.4e}")
    print(f"  Significant: {test2['significant']}")
    print(f"  Effect size: {test2['effect_size']:.4f}")

    # Compare KDE vs RDE
    test3 = compute_statistical_significance(kde_regrets, rde_regrets)
    print(f"\nKDE vs RDE (200×200):")
    print(f"  Mean KDE: {test3['mean1']:.4f}")
    print(f"  Mean RDE: {test3['mean2']:.4f}")
    print(f"  p-value:  {test3['pvalue']:.4e}")
    print(f"  Significant: {test3['significant']}")
    print(f"  Effect size: {test3['effect_size']:.4f}")


def example_6_multiple_k_values():
    """Example 6: Compare different k-fold values."""
    print("\n" + "="*80)
    print("Example 6: Comparing K-Fold Values")
    print("="*80)

    # Load results for different k values
    for k in [2, 5, 10]:
        results = load_results(
            dist_name='truncnorm',
            bids_per_auction=[10, 50],
            k_folds=k,
            methods=['ecdf', 'kde']
        )

        if results:
            plot_data = group_results_by_bids(results)
            plot_regret_densities(
                plot_data,
                save_path=f'figures/example6_truncnorm_k{k}.png',
                show=False,
                title=f'Truncated Normal (k={k})',
                wlegend=True
            )
            print(f"✓ Plot saved: figures/example6_truncnorm_k{k}.png")


def main():
    """Run all examples."""
    print("\n" + "="*80)
    print("MAPP Plot Utilities - Usage Examples")
    print("="*80)

    # Create output directory
    Path('figures').mkdir(exist_ok=True)

    try:
        example_1_basic_density_plot()
    except Exception as e:
        print(f"✗ Example 1 failed: {e}")

    try:
        example_2_boxplot()
    except Exception as e:
        print(f"✗ Example 2 failed: {e}")

    try:
        example_3_summary_table()
    except Exception as e:
        print(f"✗ Example 3 failed: {e}")

    try:
        example_4_cross_distribution()
    except Exception as e:
        print(f"✗ Example 4 failed: {e}")

    try:
        example_5_statistical_testing()
    except Exception as e:
        print(f"✗ Example 5 failed: {e}")

    try:
        example_6_multiple_k_values()
    except Exception as e:
        print(f"✗ Example 6 failed: {e}")

    print("\n" + "="*80)
    print("All examples completed!")
    print("="*80)


if __name__ == '__main__':
    main()
