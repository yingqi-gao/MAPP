from _pricing_utils import opt, dict_part, max_epc_rev
from _py_density_estimation import kde_py, rde_py
from functools import partial



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



def RSOP(bids):
    """
    Runs a random sampling optimal price auction.

    Parameters:
    - bids (dict): Bidders and corresponding bids.

    Returns:
    - Auction price (float).
    """
    group1, group2 = dict_part(bids)

    price1 = opt(group1)
    price2 = opt(group2)

    return max(price1, price2)



def RSKDE(bids, lower, upper):
    """
    Runs a random sampling kernel density estimation auction.

    Parameters:
    - bids (dict): Bidders and corresponding bids.
    - lower (float): Lower limit for bidder values and bids.
    - upper (float): Upper limit for bidder values and bids.

    Returns:
    - Auction price (float).
    - Estimated cdfs (tuple[Callable, Callable]).
    """
    # Step 1: Partition bids into two groups.
    group1, group2 = dict_part(bids)

    # Step 2: Estimate density within each group.
    cdf1 = kde_py([*group1.values()], lower, upper)
    cdf2 = kde_py([*group2.values()], lower, upper)

    # Step 3: Find the optimal estimated price for each group.
    price1 = max_epc_rev(cdf1, lower, upper)
    price2 = max_epc_rev(cdf2, lower, upper)
    
    # Return
    return max(price1, price2), (cdf1, cdf2)



def RSRDE(bids, lower, upper, *, train_hist, train_bws, method = "MLE", grid_size = 1024):
    """
    Runs a random sampling repeated density estimation auction.

    Parameters:
    - bids (dict): Bidders and corresponding bids.
    - lower (float): Lower limit for bidder values and bids.
    - upper (float): Upper limit for bidder values and bids.
    Keyword Arguments
    - train_hist (list[list[num]]): Training history, i.e., stored training observations. Each element is a numeric list storing training observations at round t.
    - train_bws (list[num]): Bandwidths selected at each round for kernel density estimation.
    - method (str): A string specifying the method to use for calculating the estimated parameters ("MLE", "MAP", "BLUP", default: "MLE").
    - grid_size (int): The number of grid points to generate for evaluating estimated density (default: 1024).

    Returns:
    - Auction price (float).
    - Estimated cdfs (tuple[Callable, Callable]).
    """ 
    # Step 1: Partition bids into two groups.
    group1, group2 = dict_part(bids)

    # Step 2: Estimate density within each group.
    cdf1 = rde_py([*group1.values()], lower, upper, train_hist = train_hist, train_bws = train_bws, method = method, grid_size = grid_size)
    cdf2 = rde_py([*group2.values()], lower, upper, train_hist = train_hist, train_bws = train_bws, method = method, grid_size = grid_size)

    # Step 3: Find the optimal estimated price for each group.
    price1 = max_epc_rev(cdf1, lower, upper)
    price2 = max_epc_rev(cdf2, lower, upper)
    
    # Return
    return max(price1, price2), (cdf1, cdf2)

