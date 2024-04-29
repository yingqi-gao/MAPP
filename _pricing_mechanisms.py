from _pricing_utils import opt, dict_part, max_epc_rev
from _py_density_estimation import kde_py, rde_testing_py
import numpy as np



def DOP(bids):
    """
    Runs a deterministic optimal price auction.

    Parameter:
    - bids (dict): Bidders and corresponding bids.

    Returns:
    - Auction price (float).
    """
    for bidder, bid in bids.items():
        # Step 1: Find the bidder-specific optimal sale price by excluding the bidder.
        bids_copy = bids.copy()
        del bids_copy[bidder]
        price_bidder = opt(bids_copy)

        # Step 2: Decide the uniform price.
        if bid >= price_bidder:
            return price_bidder



def RSOP(bids, *, random_seed):
    """
    Runs a random sampling optimal price auction.

    Parameters:
    - bids (dict): Bidders and corresponding bids.
    - random_seed (int): A random seed for all pricing mechanisms to use the same partition.

    Returns:
    - Auction price (float).
    """
    group1, group2 = dict_part(bids, random_seed)

    price1 = opt(group1)
    price2 = opt(group2)

    return max(price1, price2)



def RSKDE(bids, *, lower, upper, random_seed):
    """
    Runs a random sampling kernel density estimation auction.

    Parameters:
    - bids (dict): Bidders and corresponding bids.
    - lower (float): Lower limit for bidder values and bids.
    - upper (float): Upper limit for bidder values and bids.
    - random_seed (int): A random seed for all pricing mechanisms to use the same partition.

    Returns:
    - Auction price (float).
    - Estimated cdfs (tuple[Callable, Callable]).
    """
    # Step 1: Partition bids into two groups.
    group1, group2 = dict_part(bids, random_seed)

    # Step 2: Estimate density within each group.
    cdf1 = kde_py([*group1.values()], lower, upper)
    cdf2 = kde_py([*group2.values()], lower, upper)

    # Step 3: Find the optimal estimated price for each group.
    price1, rev1 = max_epc_rev(cdf1, lower, upper)
    price2, rev2 = max_epc_rev(cdf2, lower, upper)

    # Return
    if price1 > price2:
        return price1, cdf1
    else:
        return price2, cdf2



def RSRDE(bids, *, lower, upper, random_seed, method = "MLE", training_results):
    """
    Runs a random sampling repeated density estimation auction.

    Parameters:
    - bids (dict): Bidders and corresponding bids.
    - lower (float): Lower limit for bidder values and bids.
    - upper (float): Upper limit for bidder values and bids.
    - random_seed (int): A random seed for all pricing mechanisms to use the same partition.
    - method (str): A string specifying the method to use for calculating the estimated parameters ("MLE", "MAP", "BLUP", default: "MLE").
    - training_results (R list): Results from training on the previous auctions.

    Returns:
    - Auction price (float).
    - Estimated cdfs (tuple[Callable, Callable]).
    """ 
    # Step 1: Partition bids into two groups.
    group1, group2 = dict_part(bids, random_seed)

    # Step 2: Estimate density within each group.
    cdf1 = rde_testing_py(test_obs_at_t = [*group1.values()], 
                          method = method,
                          lower = lower,
                          training_results = training_results)
    cdf2 = rde_testing_py(test_obs_at_t = [*group2.values()], 
                          method = method,
                          lower = lower,
                          training_results = training_results)
    
    # Step 3: Find the optimal estimated price for each group.
    price1, rev1 = max_epc_rev(cdf1, lower, upper)
    price2, rev2 = max_epc_rev(cdf2, lower, upper)
    
    # Return
    if price1 > price2:
        return price1, np.array([cdf1(x) for x in np.linspace(lower, upper, num = 1024)])
    else:
        return price2, np.array([cdf2(x) for x in np.linspace(lower, upper, num = 1024)])

