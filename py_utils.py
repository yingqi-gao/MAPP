import random
from functools import partial
import scipy
import pickle


# Partition a dictionary
def dict_part(input_dict, prop):
    """
    Partitions a disctionary into two.

    Parameter:
    - input_dict (dict): Dictionary to be partitioned.
    - prop (float): Proportion of items assigned to the first partition.

    Returns:
    - New dictionary 1 (dict).
    - New dictionary 2 (dict).
    """
    keys = list(input_dict.keys())

    # Calculate the number of keys for the first partition
    dict1_size = int(len(keys) * prop)

    # Randomly sample keys for the first partition
    dict1_keys = random.sample(keys, dict1_size)

    # Create the first partition
    dict1 = {key: input_dict[key] for key in dict1_keys}

    # Create the second partition with the remaining keys
    dict2 = {key: input_dict[key] for key in keys if key not in dict1_keys}

    return dict1, dict2


# Find the optimal price - max ib_i
def opt(bids):
    """
    Finds the optimal price that maximizes revenue gained from bids received.

    Parameter:
    - bids (dict): Bidders and corresponding bids.

    Return:
    - Optimal sale price (float).
    """
    # Step 1: Sort bids in descending order
    sorted_bids = dict(sorted(bids.items(), key = lambda item: item[1], reverse = True))
    
    # Step 2: Calculate revenues based on each bid
    revenues = {bidder: (index + 1) * bid for index, (bidder, bid) in enumerate(sorted_bids.items())}

    # Step 3: Locate the maximum revenue according to the bidder
    max_bidder = max(revenues, key = revenues.get)

    # Step 4: Backtrack the optimal price
    price = bids[max_bidder]

    # Return
    return price


# Calculated the expected per capita revenue - p(1-F(p))
def get_epc_rev(price, *, value_cdf, **kwargs):
    """
    Calculates expected per capita revenue given the value cdf, i.e., p(1 - F(p)).

    Parameters:
    - price (float): Price charged to every buyer.

    Keyword Arguments
    - value_cdf (callable func): Cumulative distribution function of buyers' values.
    - **kwargs: Extra arguments passed to value_cdf.

    Return:
    - Function value (float).
    """
    return price * (1 - value_cdf(price, **kwargs))


# Find the maximum expected per capita revenue - max_p p(1-F(p))
def max_epc_rev(value_cdf, lower = 1, upper = 100, **kwargs):
    """
    Maximizes the expected per capita revenue, i.e., max_p p(1 - F(p)).

    Parameters:
    - value_pdf (callable func): Cumulative distribution function of buyers' values.
    - lower (float) : Lower limit for bidder values and bids (default: 1).
    - **kwargs: Extra arguments passed to value_cdf.

    Return:
    - Optimal price (maximum point) (float).
    - Optimal expected per capita revenue (maximum) (float).
    """
    # Step 1: Wrap get_epc_rev with the given value cdf and extra arguments if any
    wrapped_get_epc_rev = partial(get_epc_rev, value_cdf = value_cdf, **kwargs)

    # Step 2: Maximization
    price = scipy.optimize.minimize(lambda x: -wrapped_get_epc_rev(x), x0 = (lower + upper) / 2, bounds = ((lower, upper),))

    # Return
    return price.x[0]


# Transfer between the objects of interests and the files
def export_objects(t, file_name):
    """
    Exports objects of interests at a checkpoint to a file. 
    
    Parameters:
    - t (int): Current checkpoint in terms of time. 
    - file_name (str): Name of the file to store the objects at the current checkpoint.
    """
    global history, actual_revenue_history, regret_history, estimated_parameters_stored
    checkpoint = {"history": history, 
                  "actual_revenue_history": actual_revenue_history,
                  "regret_history": regret_history}
    if estimated_parameters_stored:
        global estimated_parameters_history
        checkpoint["estimated_parameters_history"] = estimated_parameters_history
        
    with open(file_name, "wb") as file:
        pickle.dump(checkpoint, file)
        print(f"Checkpoint at time {t} saved to {file_name}")

def import_objects(file_name):
    """
    Imports objects of interests from a file.
    
    Parameter:
    - file_name (str): Name of the file to import the objects from.
    
    Return: 
    A dictionary of objects of interests.
    """
    with open(file_name, "rb") as file:
        dict_objects = pickle.load(file)

    print(len(dict_objects["regret_history"]))
    
    return dict_objects