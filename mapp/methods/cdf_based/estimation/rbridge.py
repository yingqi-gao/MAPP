"""
R bridge for CDF estimation functions.
Simple setup that returns R functions ready to use.

Prerequisites: Run ./setup.sh first to install R packages and set up environment.

Internal module - use high-level APIs like cdf_based_pricing() or get_trained_rde_model() instead.
"""

import rpy2.robjects as ro
from rpy2.robjects import numpy2ri
from pathlib import Path


def _setup_rbridge():
    """
    Initialize R bridge and return all CDF estimation functions.

    Internal function - not meant for direct use.
    Returns: (ecdf_r, kde_cdf_r, train_rde_r, rde_cdf_r)
    """
    # Enable automatic numpy-R conversion
    numpy2ri.activate()

    # Load CDF functions from cdf.R
    cdf_r_file = Path(__file__).parent / "cdf.R"
    if not cdf_r_file.exists():
        raise FileNotFoundError(f"cdf.R not found at {cdf_r_file}")

    ro.r.source(str(cdf_r_file))

    # Return R function references
    return (
        ro.r["ecdf"],  # Built-in R empirical CDF function
        ro.globalenv["kde_cdf_r"],
        ro.globalenv["train_rde_r"],
        ro.globalenv["rde_cdf_r"]
    )


if __name__ == "__main__":
    print("Testing R bridge setup...")
    try:
        ecdf_r, kde_cdf_r, train_rde_r, rde_cdf_r = _setup_rbridge()
        print("✅ R bridge initialized successfully")

        # Test KDE function
        import numpy as np
        sample_data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        kde_cdf = kde_cdf_r(sample_data, 0.5, 5.5, 100)
        test_value = ro.r('function(f) f(3.0)')(kde_cdf)[0]
        print(f"✅ KDE CDF test: F(3.0) = {test_value:.3f}")

        # Test empirical CDF
        emp_cdf = ecdf_r(sample_data)
        emp_value = ro.r('function(f) f(3.0)')(emp_cdf)[0]
        print(f"✅ Empirical CDF test: F(3.0) = {emp_value:.3f}")

    except Exception as e:
        print(f"❌ Setup failed: {e}")
        import traceback
        traceback.print_exc()