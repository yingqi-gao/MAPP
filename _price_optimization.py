import os
from rpy2.robjects import numpy2ri
import rpy2.robjects as ro
from functools import partial
import scipy.stats as stats
from scipy.optimize import minimize_scalar


# Set the R_HOME address
os.environ["R_HOME"] = "/Library/Frameworks/R.framework/Resources"

# Activate automatic conversion between R and Python (NumPy)
numpy2ri.activate()

# Source the R script (load all functions from the R file)
ro.r["source"]("_density_estimation.r")


# Access the R functions for density estimation from the R environment
kde = ro.globalenv["kde_r"]
rde = ro.globalenv["rde_r"]
ecdf = ro.globalenv["ecdf_r"]


def get_objective(p, cdf):
    """
    A small helper function that calculates the expected per capita revenue.

    Args:
        p (float): The given price, at which the expected per capita revenue is calculated.
        cdf (function): The given Cumulative Distribution Function (CDF), for which the expected per capita revenue is calculated.

    Returns:
        float: The negative expected per capita revenue for the given Cumulative Distribution Function (CDF) at the given price p.
    """
    return -p * (1 - cdf(p))


def get_optimals(cdfs, lower, upper):
    """
    For each given Cumulative Distribution Function (CDF) F(.), get the optimal price that maximizes the expected per capita revenue p(1-F(p)) and the corresponding expected per capita revenue.

    Args:
        cdfs (list of functions): A list of Python functions, each of which is a Cumulative Distribution Function (CDF) that can be evaluated at any given point.
        lower (float): A numeric value specifying the lower bound of the common interval to which the optimizer is restricted.
        upper (float): A numeric value specifying the upper bound of the common interval to which the optimizer is restricted.

    Returns:
        list of tuple: A list where each tuple contains:
            - optimal_price (float): The price that maximizes the expected per capita revenue corresponding to a given CDF.
            - optimal_revenue (float): The expected per capita revenue at the optimal price.

    Example:
        >>> # Generate some CDFs
        >>> samples = truncnorm.rvs(a=-1, b=1, loc=0, scale=1, size=(50, 200))
        >>> cdfs = ecdf(samples)
        >>> # Run the function
        >>> results = get_optimals(cdfs, -1, 1)
        >>> print(results[randrange(50)])
        (array([0.49708447]), array([0.09941689]))
    """
    results = []

    for cdf in cdfs:
        optimals = minimize_scalar(
            partial(get_objective, cdf=cdf), bounds=(lower, upper), method="bounded"
        )
        results.append((optimals.x, -optimals.fun))

    return results


def get_ideals(dist_name, lower, upper, params_list):
    """
    For each distribution of given parameter(s), get the ideal price that maximizes the expected per capita revenue p(1-F(p)) and the corresponding expected per capita revenue.

    Args:
        dist_name (str): The name of the distribution (e.g., "uniform", "truncnorm", "truncexpon").
        lower (float): A numeric value specifying the lower bound of the common support of the distributions.
        upper (float): A numeric value specifying the upper bound of the common support of the distributions.
        params_list (list of dict): A list where each element is a dictionary containing the parameters for one application of the distribution.

    Returns:
        list of tuple: A list where each tuple contains:
            - ideal_price (float): The price that maximizes the expected per capita revenue corresponding to a given distribution.
            - ideal_revenue (float): The expected per capita revenue at the ideal price.

    Examples:
        >>> # Generate a random list of parameters
        >>> params_list = [
                {
                    "a": -1,
                    "b": 1,
                    "loc": uniform(-1, 1),
                    "scale": uniform(0, 2)
                }
            for _ in range(50)
            ]
        >>> # Run the function
        >>> results = get_ideals("truncnorm", -1, 1, params_list)
        >>> print(results[randrange(50)])
    """
    results = []

    # Step 1: Get the distribution function from scipy.stats based on dist_name.
    dist_func = getattr(stats, dist_name, None)
    if dist_func is None:
        raise ValueError(f"Distribution '{dist_name}' not found in scipy.stats")

    # Step 2: Calculate the ideals
    for params in params_list:
        cdf = partial(dist_func.cdf, **params)
        ideals = minimize_scalar(
            partial(get_objective, cdf=cdf), bounds=(lower, upper), method="bounded"
        )
        results.append((ideals.x, -ideals.fun))

    return results


if __name__ == "__main__":
    print("running tests...")
    from random import uniform, randrange

    train_samples = stats.truncnorm.rvs(a=-1, b=1, loc=0, scale=1, size=(50, 200))
    test_samples = stats.truncnorm.rvs(a=-1, b=1, loc=0, scale=1, size=(200, 10))

    # test ecdf
    print(ecdf(train_samples)[randrange(50)](0.32))
    print(stats.truncnorm.cdf(0.32, a=-1, b=1, loc=0, scale=1))

    # test kde
    print(kde(train_samples, -1, 1, 1024)[randrange(50)](0.32))
    print(stats.truncnorm.cdf(0.32, a=-1, b=1, loc=0, scale=1))

    # test rde
    print(rde(train_samples, test_samples, -1, 1, 1024)[randrange(200)](0.32))
    print(stats.truncnorm.cdf(0.32, a=-1, b=1, loc=0, scale=1))

    # test get_optimals
    cdfs = ecdf(train_samples)
    results = get_optimals(cdfs, -1, 1)
    print(results[randrange(50)])

    # test get_ideals
    params_list = [
        {"a": -1, "b": 1, "loc": uniform(-1, 1), "scale": uniform(0, 2)}
        for _ in range(50)
    ]
    results = get_ideals("truncnorm", -1, 1, params_list)
    print(results[randrange(50)])
