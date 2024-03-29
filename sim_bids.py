import random
import scipy
from py_utils import get_epc_rev, max_epc_rev
from r2py_utils import get_bw
import pickle


# Simulate a single-round auction.
def simulate_bids(min_num_bidders = 10, max_num_bidders = 100,
                  dist = "uniform", dist_params = {}, 
                  lower = 1, upper = 200):
    """
    Simulates bids received in a single-round auction.
    
    Parameters:
    - min_num_bidders (int): Minimum number of bidders participating in each round of auctions (default: 10).
    - max_num_bidders (int): Maximum number of bidders participating in each round of auctions (default: 100). 
    - dist (str): Type of bidders' value distribution in each round of auctions ("uniform", "normal", "pareto", "exponential").
    - dist_params (dict): Parameters of bidders' value distribution in each round of auctions.
    - lower (float): Lower limit for bidders' values/bids in each round of auctions (default: 0).
    - upper (float): Upper limit for bidders' values/bids in each round of auctions (default: 100).
    
    Return: 
    - A dictionary of bidders and their bids in a single-round auction.
    - Number of bidders in the auction.
    - Bidders' value distribution
    """
    # Step 1: Generate the number of bidders.
    num_bidders = random.randint(min_num_bidders, max_num_bidders)
    
    # Step 2: Define bidders.
    bidders = [f"Bidder {i+1}" for i in range(num_bidders)]

    # Step 3: Assign values.
    ## Check if distribution parameters are specified.
    if len(dist_params) == 0:
        dist_params = {
            "mean": random.uniform(1e-10, 2 * upper),
            "sd": random.uniform(1e-10, upper),
            "scale": random.uniform(1e-10, 2 * upper),
            "location": lower
        }
    ## Convert distribution parameters to scipy arguments.
    distributions = {
        "uniform": [
            scipy.stats.uniform,
            {"loc": lower, 
             "scale": upper - lower}
        ],
        "normal": [
            scipy.stats.truncnorm, 
            {"a": (lower - dist_params["mean"]) / dist_params["sd"], 
             "b": (upper - dist_params["mean"]) / dist_params["sd"], 
             "loc": dist_params["mean"], 
             "scale": dist_params["sd"]}
        ],
        "exponential": [
            scipy.stats.truncexpon,
            {"b": (upper - dist_params["location"]) / dist_params["scale"], 
             "loc": dist_params["location"], 
             "scale": dist_params["scale"]}
        ]
    }
    ## Check if distribution specified is supported.
    if dist not in distributions:
        raise ValueError("Invalid distribution specified.")
    ## Generate the bidders value distribution.
    bidders_value_dist = distributions[dist]
    ## Generate values.
    bidders_values = {bidder: bidders_value_dist[0].rvs(**bidders_value_dist[1]) for bidder in bidders}

    # Step 4: Bidding Process.
    bids = bidders_values # assume truthtelling
    
    # Return
    return bids, num_bidders, bidders_value_dist


# Simulate a multi-round auction.
def simulate_bids_history(num_rounds = 200, 
                          min_num_bidders = 10, max_num_bidders = 100, 
                          dist = "uniform", dist_params = {}, 
                          lower = 1, upper = 200, upper_float = False):
    """
    Simulates bids received in repeated auctions.
    
    Parameters:
    - num_rounds (int): Total number of rounds of auctions (default: 200). 
    - min_num_bidders (int): Minimum number of bidders participating in each round of auctions (default: 10).
    - max_num_bidders (int): Maximum number of bidders participating in each round of auctions (default: 100).
    - dist (str): Type of bidders' value distribution in each round of auctions ("uniform", "normal", "pareto", "exponential").
    - dist_params (dict): Parameters of bidders' value distribution in each round of auctions.
    - lower (float): Lower limit for bidders' values/bids in each round of auctions (default: 0).
    - upper (float): Upper limit for bidders' values/bids in each round of auctions (default: 200).
    - upper_float (bool): Whether the upper bound of all the values or bids changes from round to round (default: False).
    
    Return: A dictionary of 
    - a list (history) of bids received,
    - a list (history) of numbers of bidders,
    - a list (history) of bidders' value distributions,
    - a list (history) of ideal prices,
    - a list (history) of ideal revenues,
    - a list (history) of bandwidths for kernel density estimation,
    - an optional list (history) of upper bounds,
    - an optional list (history) of training bids bounds,
    - an optional list (history) of training bandwidths bounds,
    in each round in the repeated auctions.
    """
    # Step 1: Create empty lists for storage.
    bids_history = []
    num_bidders_history = []
    bidders_value_dist_history = []
    ideal_price_history = []
    ideal_revenue_history = []
    kde_bw_history = []
    if upper_float:
        upper_history = []
        train_bid_history = []
        train_bw_history = []
    
    for i in range(num_rounds):
        # Step 2: Start simulating the bids.
        ## If upper floats from round to round, simulate the upper for the current round first.
        if upper_float:
            upper_t = random.uniform(0.5 * upper, 2 * upper)
            upper_history.append(upper_t)
            upper_sim = upper_t
        else:
            upper_sim = upper
        ## Simulate bids.
        bids, num_bidders, bidders_value_dist = simulate_bids(min_num_bidders, max_num_bidders, 
                                                              dist, dist_params, 
                                                              lower, upper_sim)
        bids_history.append(bids)
        num_bidders_history.append(num_bidders)
        bidders_value_dist_history.append(bidders_value_dist)
        ## If upper floats, scale bids to fall within a common support for repeated density estimation.
        if upper_float:
            train_bids = bids.copy()
            for bidder in train_bids.keys():
                train_bids[bidder] = lower + (train_bids[bidder] - lower) / (upper_t - lower) * (upper - lower)
                train_bid_history.append(train_bids)

        # Step 3: Calculate and record the ideal price based on the true distribution.
        ideal_price = max_epc_rev(bidders_value_dist[0].cdf, lower = lower, upper = upper_sim, **bidders_value_dist[1])
        ideal_price_history.append(ideal_price)
        
        # Step 4: Calculate and record the ideal expected per capita revenue.
        ideal_epc_revenue = get_epc_rev(ideal_price, value_cdf = bidders_value_dist[0].cdf, **bidders_value_dist[1])
        ideal_revenue_history.append(ideal_epc_revenue)
        
        # Step 5: Calculate and record the bandwidth for kde.
        kde_bw = get_bw([*bids.values()])
        kde_bw_history.append(kde_bw)
        if upper_float:
            train_bw = get_bw([*train_bids.values()])
            train_bw_history.append(train_bw)
    
    
    history = {"bids": bids_history,
               "num_bidders": num_bidders_history,
               "bidders_value_dists": bidders_value_dist_history,
               "ideal_prices": ideal_price_history, 
               "ideal_revenues": ideal_revenue_history,
               "kde_bws": kde_bw_history}
    if upper_float:
        history["uppers"] = upper_history
        history["train_bids"] = train_bid_history
        history["train_bws"] = train_bw_history
    
    # Return
    return history


# Start the actual simulations.
## Global variables.
lower = 1
upper = 10

## Number of bidders limited to 100 maximum.
# Upper does NOT float.
history_small_uniform = simulate_bids_history(dist = "uniform", upper = upper)
history_small_normal = simulate_bids_history(dist = "normal", upper = upper)
history_small_exponential = simulate_bids_history(dist = "exponential", upper = upper)
# Upper floats.
history_float_small_uniform = simulate_bids_history(dist = "uniform", upper = upper, upper_float = True)
history_float_small_normal = simulate_bids_history(dist = "normal", upper = upper, upper_float = True)
history_float_small_exponential = simulate_bids_history(dist = "exponential", upper = upper, upper_float = True)

## Number of bidders limited to 10,000 maximum.
# Upper does NOT float.
history_large_uniform = simulate_bids_history(max_num_bidders = 10000, dist = "uniform", upper = upper)
history_large_normal = simulate_bids_history(max_num_bidders = 10000, dist = "normal", upper = upper)
history_large_exponential = simulate_bids_history(max_num_bidders = 10000, dist = "exponential", upper = upper)
# Upper floats.
history_float_large_uniform = simulate_bids_history(max_num_bidders = 10000, dist = "uniform", upper = upper, upper_float = True)
history_float_large_normal = simulate_bids_history(max_num_bidders = 10000, dist = "normal", upper = upper, upper_float = True)
history_float_large_exponential = simulate_bids_history(max_num_bidders = 10000, dist = "exponential", upper = upper, upper_float = True)

## Store all histories generated for easy references.
histories_fix = {"small": 
                 {"uniform": history_small_uniform, 
                  "normal": history_small_normal, 
                  "exponential": history_small_exponential},               
                 "large": 
                 {"uniform": history_large_uniform, 
                  "normal": history_large_normal, 
                  "exponential": history_large_exponential}}
histories_float = {"small": 
                   {"uniform": history_float_small_uniform, 
                    "normal": history_float_small_normal, 
                    "exponential": history_float_small_exponential},               
                   "large": 
                   {"uniform": history_float_large_uniform, 
                    "normal": history_float_large_normal, 
                    "exponential": history_float_large_exponential}}
histories = {"fix": histories_fix,
             "float": histories_float}
with open("histories.pkl", "wb") as file:
    pickle.dump(histories, file)