from pricing import RSOP, DOP, RSDE
from _pricing_utils import dict_part, get_epc_rev, export_objects


def simulate_auction(pricing = "RSOP", prop = 0.5, repeated = True, t = 100, upper_float = False):
    """
    Simulates a single-round auction.

    Parameters:
    - pricing (str): Mechanism for deciding the auction price ("RSOP", "DOP", "RSDE").
    - prop (float): Proportion of items assigned to the first partition (default: 0.5).
    - repeated (bool): Whether to use historcial data in density estimation if any (default: True). 
    - t (int): At which round the auction is implemented (only matters to repeated density estimation, default: 100).
    - upper_float (bool): Whether the upper bound of all the values or bids changes from round to round (default: False).

    Side Effects: 
    - Records actual revenue.
    - Records per capita regret.
    """
    # Step 1: Retrieve bids at the current round. 
    global history
    bids = history["bids"][t]

    # Step 2: Check if the pricing mechanism specified is supported and make some preparation.
    pricing_mechanisms = {
        "RSOP": RSOP,
        "DOP": DOP,
        "RSDE": RSDE
    }
    if pricing not in pricing_mechanisms:
        raise ValueError("Invalid pricing mechanism specified.")
    
    # Step 3: Calculate the uniform price and the results of the auction. 
    if pricing == "DOP":
        price, winners = DOP(bids)
    else:
        group1, group2 = dict_part(bids, prop)
        price, results = pricing_mechanisms[pricing](group1, group2, repeated = repeated, t = t, upper_float = upper_float)

    # Step 4: Calculate and record the actual expected per capita revenue.
    bidders_value_dist = history["bidders_value_dists"][t]
    actual_epc_revenue = get_epc_rev(price, value_cdf = bidders_value_dist[0].cdf, **bidders_value_dist[1])
    global actual_revenue_history
    actual_revenue_history.append(actual_epc_revenue)

    # Step 5: Calculate and record the expected per capita regret.
    ideal_epc_revenue = history["ideal_revenues"][t]
    epc_regret = ideal_epc_revenue - actual_epc_revenue
    global regret_history
    regret_history.append(epc_regret)


def simulate_loop(history, pricing = "RSOP", prop = 0.5, repeated = True, upper_float = False, file_name = ".pkl"):
    """
    Simulates a multi-round auction.

    Parameters:
    - history (dict): History about bidders in multiple rounds of auctions.
    - pricing (str): Mechanism for deciding the auction price ("RSOP", "DOP", "RSDE").
    - prop (float): Proportion of items assigned to the first partition (default: 0.5).
    - repeated (bool): Whether to use historcial data in density estimation if any (default: True). 
    - upper_float (bool): Whether the upper bound of all the values or bids changes from round to round (default: False).
    - file_name (str): Name of the file to store the objects at the current checkpoint.

    Side Effects: 
    - Export objects of interests to the file specified.
    """
    # Step 1: Create empty lists to store future objects of interest.
    global actual_revenue_history, regret_history, estimated_parameters_history, estimated_parameters_stored
    actual_revenue_history, regret_history, estimated_parameters_history = [], [], []
    estimated_parameters_stored = False

    # Step 2: Start running auctions.
    ## Total number of rounds to run. 
    num_rounds = len(history["num_bidders"])
    for t in range(num_rounds):
        simulate_auction(pricing, prop, repeated, t, upper_float)
      
        ## Checkpoint every 10 rounds.
        if t%10 == 9:
            export_objects(t, file_name)