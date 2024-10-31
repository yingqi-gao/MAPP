from random import uniform
import scipy.stats as stats
from _price_optimization import kde, rde, ecdf, get_optimals, get_ideals
import numpy as np


def simulate_value_dists(N, dist_name, lower, upper):
    """
    Generate N value distributions drawn from a specified distribution type ("truncnorm", "truncexpon", or "truncpareto") with a common support (lower, upper).

    Args:
        N (int): An integer specifying the number of distributions to be generated.
        dist_name (str): The name of the distributions. Options include "truncnorm", "truncexpon", and "truncpareto".
        lower (float): A numeric value specifying the lower bound of the common support of the distributions.
        upper (float): A numeric value specifying the upper bound of the common support of the distributions.

    Returns:
        list of dict: A list where each dictionary contains the parameters of one distribution.
    """

    # Convert bounds to standard normal units
    if dist_name == "truncnorm":
        params_list = []
        for _ in range(N):
            width = upper - lower
            sigma = uniform(0.15 * width, 0.3 * width)
            mid = (upper + lower) / 2
            mu = uniform(mid - sigma, mid + sigma)
            params = {
                "a": (lower - mu) / sigma,
                "b": (upper - mu) / sigma,
                "loc": mu,
                "scale": sigma,
            }
            params_list.append(params)

    elif dist_name == "truncexpon":
        params_list = []
        for _ in range(N):
            width = upper - lower
            sigma = uniform(0.15 * width, 0.3 * width)
            mid = (upper + lower) / 2
            inverse_rate = uniform(mid - sigma, mid + sigma)
            b = (upper - lower) / inverse_rate
            params = {"b": b, "loc": lower, "scale": inverse_rate}
            params_list.append(params)

    elif dist_name == "truncpareto":
        params_list = []
        for _ in range(N):
            shape = uniform(1.15, 1.3)
            loc = lower / 10
            scale = lower - loc
            params = {
                "b": shape,
                "c": (upper - loc) / scale,
                "loc": loc,
                "scale": scale,
            }
            params_list.append(params)

    return params_list


def simulate_bids(
    dist_name, lower, upper, max_N=200, max_n=200, max_N_train=200, max_n_train=200
):
    """
    Generate bids (or values) from each value distribution in a specified list.

    Args:
        dist_name (str): The name of the distributions. Options include "truncnorm", "truncexpon", and "truncpareto".
        lower (float): A numeric value specifying the lower bound of the common support of the distributions.
        upper (float): A numeric value specifying the upper bound of the common support of the distributions.
        params_list (list of dict): A list where each element is a dictionary containing the parameters for one application of the distribution.
        max_N (int, optional): An integer specifying the maximum number of auction rounds to generate for tetsing. Defaults to 200.
        max_n (int, optional): An integer specifying the maximum number of bids to generate for testing. Defaults to 200.
        max_N_train (int, optional): An integer specifying the maximum number of auction rounds to generate for training the RDE method. Defaults to 200.
        max_n_train (int, optional): An integer specifying the maximum number of bids  to generate for training the RDE method. Defaults to 200.

    Returns:
        dict: A dictionary that contains:
            - dist_name (str): The name of the distributions.
            - params_list (list of dict): A list where each element is a dictionary containing the parameters for one application of the distribution.
            - bids (numpy.ndarray): An array of max_N rows, each of which contains max_n bids generated for testing.
            - train_bids (numpy.ndarray): A array of max_N_train rows, each of which contains max_n_train bids generated for training the RDE method.
            - ideals (list of tuples): A list of tuples, each of which contains the ideal price and the ideal expected per capita revenue.
    """
    results = {"dist_name": dist_name}

    # Step 1: Generate two lists of parameters for training and testing, respectively.
    train_params_list = simulate_value_dists(max_N_train, dist_name, lower, upper)
    params_list = simulate_value_dists(max_N, dist_name, lower, upper)
    results["params_list"] = params_list

    # Step 2: Get the distribution function from scipy.stats based on dist_name.
    dist_func = getattr(stats, dist_name, None)
    if dist_func is None:
        raise ValueError(f"Distribution '{dist_name}' not found in scipy.stats")

    # Step 3: Generate bids according to the list of parameters.
    results["bids"] = np.array(
        [dist_func.rvs(**params, size=max_n) for params in params_list]
    )
    # print(results["bids"].min(), results["bids"].max())
    results["train_bids"] = np.array(
        [dist_func.rvs(**params, size=max_n_train) for params in train_params_list]
    )
    # print(results["train_bids"].min(), results["train_bids"].max())

    # Check if distributions are truncated as expected:
    # if not np.all((results["bids"] >= lower) & (results["bids"] <= upper)) and not np.all((results["train_bids"] >= lower) & (results["train_bids"] <= upper)):
    #     raise ValueError("Parameters set for distributions are incorrect!")

    # Step 4: Calculate the ideal price and the ideal expected per capita revenue according to the list of parameters.
    results["ideals"] = get_ideals(dist_name, lower, upper, params_list)

    # Return the dictionary
    return results


def simulate_regrets(bids_dict, lower, upper, method, N, n, N_train=None, n_train=None):
    """
    Simulate N rounds of auctions. Each aution receives n bids. Apply the Max-Price ADPP mechanism with an initial auction using eCDF, KDE, or RDE to estimate the optimal price, and compute the regret.

    Args:
        bids_dict (dict): A dictionary that contains:
                            - dist_name (str): The name of the distributions.
                            - params_list (list of dict, optional): A list where each element is a dictionary containing the parameters for one application of the distribution.
                            - true_cdfs (list of functions, optional): A list of Python functions, each of which is a true Cumulative Distribution Function (CDF) that can be evaluated at any given point.
                            - bids (numpy.ndarray): An array of max_N rows, each of which contains max_n bids generated for testing.
                            - train_bids (numpy.ndarray): A array of max_N_train rows, each of which contains max_n_train bids generated for training the RDE method.
                            - ideals (list of tuples): A list of tuples, each of which contains the ideal price and the ideal expected per capita revenue.
        lower (float): A numeric value specifying the lower bound of the common support of the distributions.
        upper (float): A numeric value specifying the upper bound of the common support of the distributions.
        method (str): The method of the initial auction to be used for price optimization. Options include "ecdf", "kde", and "rde".
        N (int): An integer specifying the number of auction rounds to be generated.
        n (int): An integer specifying the number of bids received at each auction round.
        N_train (int, optional): An integer specifying the number of past auction rounds available for training with the RDE method.
        n_train (int, optional): An integer specifying the number of bids received at each past auction round for training with the RDE method.

    Returns:
        list: A list of regrets involved in the N rounds of auctions.
    """
    # Check if both keys are present
    if "params_list" in bids_dict and "true_cdfs" in bids_dict:
        raise ValueError(
            "Please provide either 'params_list' or 'true_cdfs', not both."
        )
    # Raise an error if neither key is present
    elif "params_list" not in bids_dict and "true_cdfs" not in bids_dict:
        raise ValueError(
            "The dictionary must contain either 'params_list' or 'true_cdfs'."
        )

    regrets = []

    # Handle the case where only 'params_list' is provided
    if "params_list" in bids_dict:
        params_list = bids_dict["params_list"]

    # Handle the case where only 'true_cdfs' is provided
    elif "true_cdfs" in bids_dict:
        true_cdfs = bids_dict["true_cdfs"]

    # Step 1: Extract the parameters to be used.
    samples = bids_dict["bids"][:N, :n]

    # Step 2: Construct the CDFs using the specified method.
    if method == "ecdf":
        cdfs = ecdf(samples)
    elif method == "kde":
        cdfs = kde(samples, lower, upper)
    elif method == "rde":
        train_samples = bids_dict["train_bids"][:N_train, :n_train]
        cdfs = rde(train_samples, samples, lower, upper)
    else:
        raise ValueError(f"Method '{method}' not supported.")

    # Step 3: Calculate the optimal prices and the corresponding expected per capita revenues.
    if "params_list" in bids_dict:
        optimals = get_optimals(
            cdfs, bids_dict["dist_name"], lower, upper, params_list=params_list[:N]
        )
    elif "true_cdfs" in bids_dict:
        optimals = get_optimals(
            cdfs, bids_dict["dist_name"], lower, upper, true_cdfs=true_cdfs[:N]
        )

    # Step 4: Calculate the regrets.
    optimal_revenues = np.hstack([optimal[1] for optimal in optimals])
    ideal_revenues = np.hstack([ideal[1] for ideal in bids_dict["ideals"]])
    regrets = np.array([ideal_revenues[i] - optimal_revenues[i] for i in range(N)])
    # # Check if regrets are nonnegative as expected:
    # if not np.all(regrets >= 0):
    #     print(regrets[np.where(regrets < 0)])
    #     raise ValueError("Regrets cannot be negative!")

    # Return the list of regrets.
    return regrets


if __name__ == "__main__":
    print("running tests...")
    from random import randrange

    # test simulate_value_dists
    params_list = simulate_value_dists(N=200, dist_name="truncnorm", lower=1, upper=10)
    print(params_list[randrange(200)])
    params_list = simulate_value_dists(N=200, dist_name="truncexpon", lower=1, upper=10)
    print(params_list[randrange(200)])
    params_list = simulate_value_dists(
        N=200, dist_name="truncpareto", lower=1, upper=10
    )
    print(params_list[randrange(200)])

    # test simulate_bids
    bids_dict = simulate_bids(
        "truncnorm",
        lower=0.9,
        upper=10.1,
        max_N=200,
        max_n=200,
        max_N_train=200,
        max_n_train=200,
    )
    print(bids_dict["dist_name"])
    print(bids_dict["params_list"][randrange(200)])
    print(bids_dict["bids"][randrange(200)])
    print(bids_dict["train_bids"][randrange(200)])
    print(bids_dict["ideals"][randrange(200)])
    bids_dict = simulate_bids(
        "truncexpon",
        lower=0.9,
        upper=10.1,
        max_N=200,
        max_n=200,
        max_N_train=200,
        max_n_train=200,
    )
    print(bids_dict["dist_name"])
    print(bids_dict["params_list"][randrange(200)])
    print(bids_dict["bids"][randrange(200)])
    print(bids_dict["train_bids"][randrange(200)])
    print(bids_dict["ideals"][randrange(200)])
    bids_dict = simulate_bids(
        "truncpareto",
        lower=0.9,
        upper=10.1,
        max_N=200,
        max_n=200,
        max_N_train=200,
        max_n_train=200,
    )
    print(bids_dict["dist_name"])
    print(bids_dict["params_list"][randrange(200)])
    print(bids_dict["bids"][randrange(200)])
    print(bids_dict["train_bids"][randrange(200)])
    print(bids_dict["ideals"][randrange(200)])

    # test simulate_regrets
    results = simulate_regrets(
        bids_dict, 1, 10, "ecdf", N=100, n=10, N_train=None, n_train=None
    )
    print(results[randrange(100)])
    results = simulate_regrets(
        bids_dict, 1, 10, "kde", N=100, n=10, N_train=None, n_train=None
    )
    print(results[randrange(100)])
    results = simulate_regrets(bids_dict, 1, 10, "rde", N=100, n=10, N_train=100, n_train=100)
    print(results[randrange(100)])
