from _pricing_utils import opt, max_epc_rev
from _py_density_estimation import kde_py, rde_py
from functools import partial



def DOP(bids):
    """
    Runs a deterministic optimal price auction.

    Parameter:
    - bids (dict): Bidders and corresponding bids.

    Returns:
    - Auction price (float).
    - Winners (list[str]).
    """
    winners = []
    price = 0
    for bidder, bid in bids.items():
        # Step 1: Find the bidder-specific optimal sale price by excluding the bidder.
        bids_copy = bids.copy()
        del bids_copy[bidder]
        price_bidder = opt(bids_copy)

        # Step 2: Decide winners.
        if bid >= price_bidder:
            if price == 0:
                price = price_bidder
            winners.append(bidder)

    # Return
    return price, winners



def RSOP(group1, group2):
    """
    Runs a random sampling optimal price auction.

    Parameters:
    - group1 (dict): Bidders and corresponding bids in group 1.
    - group2 (dict): Bidders and corresponding bids in group 2.

    Returns:
    - Auction price (float).
    - Auction results (dict).
    """
    # Step 1: Find the optimal sale price of each group.
    price1 = opt(group1)
    price2 = opt(group2)
    price = max(price1, price2)

    # Step 2: Decide winners in the other group.
    winners1 = [bidder for bidder, bid in group1.items() if bid >= price2]
    winners2 = [bidder for bidder, bid in group2.items() if bid >= price1]
    
    # Step 3: Store auction results.
    results = {"group1": {"bidders": group1, "price": price2, "winners": winners1},
               "group2": {"bidders": group2, "price": price1, "winners": winners2}}
    
    # Return
    return price, results



def RSKDE(group1, group2, lower, upper):
    """
    Runs a random sampling kernel density estimation auction.

    Parameters:
    - group1 (dict): Bidders and corresponding bids in group 1.
    - group2 (dict): Bidders and corresponding bids in group 2.
    - lower (float): Lower limit for bidder values and bids.
    - upper (float): Upper limit for bidder values and bids.

    Returns:
    - Auction price (float).
    - Auction results (dict).
    """
    # Step 1: Estimate density within each group.
    cdf1 = kde_py([*group1.values()], lower, upper)
    cdf2 = kde_py([*group2.values()], lower, upper)

    # Step 2: Find the optimal estimated price for each group.
    price1 = max_epc_rev(cdf1, lower, upper)
    price2 = max_epc_rev(cdf2, lower, upper)
    price = max(price1, price2)

    # Step 3: Decide winners in the other group.
    winners1 = [bidder for bidder, bid in group1.items() if bid >= price2]
    winners2 = [bidder for bidder, bid in group2.items() if bid >= price1]
    
    # Step 4: Store auction results.
    results = {"group1": {"bidders": group1, "price": price2, "winners": winners1},
               "group2": {"bidders": group2, "price": price1, "winners": winners2}}
    
    # Return
    return price, results



def RSRDE(group1, group2, lower, upper, *, train_hist, train_bws, method = "MLE", grid_size = 1024):
    """
    Runs a random sampling repeated density estimation auction.

    Parameters:
    - group1 (dict): Bidders and corresponding bids in group 1.
    - group2 (dict): Bidders and corresponding bids in group 2.
    - lower (float): Lower limit for bidder values and bids.
    - upper (float): Upper limit for bidder values and bids.
    Keyword Arguments
    - train_hist (list[list[num]]): Training history, i.e., stored training observations. Each element is a numeric list storing training observations at round t.
    - train_bws (list[num]): Bandwidths selected at each round for kernel density estimation.
    - method (str): A string specifying the method to use for calculating the estimated parameters ("MLE", "MAP", "BLUP", default: "MLE").
    - grid_size (int): The number of grid points to generate for evaluating estimated density (default: 1024).

    - repeated (bool): Whether to use historcial data for density estimation (default: True). 
    - t (int): At which round the auction is implemented (only matters to repeated density estimation, default: 100).
    - upper_float (bool): Whether the upper bound of all the values or bids changes from round to round (default: False).

    Returns:
    - Auction price (float).
    - Winners (list of str).
    """ 
    # Step 1: Retrieve the global objects. 
    global history, lower, upper

    # Step 2: Get the correct upper value for estimation.
    if upper_float and not repeated:
        upper_t = history["uppers"][t]
        upper_est = upper_t
    else:
        upper_est = upper

    # Step 3: Estimate density using bids in each group respectively.
    ## If the upper floats, use scaled bids for repeated density estimation.
    if upper_float and repeated:
        # test 
        test_bids = history["train_bids"][t]
        test_obs1 = [test_bids[key] for key in [*group1.keys()]]
        test_obs2 = [test_bids[key] for key in [*group2.keys()]]
        # train
        train_hist = [[*bids_dict.values()] for i, bids_dict in enumerate(history["train_bids"]) if i < t]
        train_bws = history["train_bws"][:t]
    else: 
        # test
        test_obs1 = [*group1.values()]
        test_obs2 = [*group2.values()]
        # train
        train_hist = [[*bids_dict.values()] for i, bids_dict in enumerate(history["bids"]) if i < t]
        train_bws = history["kde_bws"][:t]
    ## Estimate density for each group.
    est_cdf1 = density_est(train_hist = train_hist, train_bws = train_bws, test_obs_at_t = test_obs1, lower = lower, upper = upper_est, repeated = repeated)
    est_cdf2 = density_est(train_hist = train_hist, train_bws = train_bws, test_obs_at_t = test_obs2, lower = lower, upper = upper_est, repeated = repeated)

    # Step 4: Find the optimal price from the estimated cdf.
    price1 = max_epc_rev(est_cdf1, lower = lower, upper = upper_est)
    price2 = max_epc_rev(est_cdf2, lower = lower, upper = upper_est)
    price = max(price1, price2)
    ## If scaled bids used, scale back to the actual support.
    if upper_float and repeated:
        price = lower + (price - lower) / (upper - lower) * (upper_t - lower)
        
    # Step 5: Decide winners in the other group.
    winners2 = [bidder for bidder, bid in group2.items() if bid >= price1]
    winners1 = [bidder for bidder, bid in group1.items() if bid >= price2]
    
    # Step 6: Store auction results.
    results = {"group1": {"bidders": group1, "price": price2, "winners": winners1},
               "group2": {"bidders": group2, "price": price1, "winners": winners2}}

    # Return
    return price, results