import random
from functools import partial
import scipy
import pickle



# Find the optimal price - max ib_i
def opt(bids: dict) -> float:
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



# Partition a dictionary
def dict_part(input_dict, prop = 0.5):
    """
    Partitions a disctionary into two.

    Parameter:
    - input_dict (dict): Dictionary to be partitioned.
    - prop (float): Proportion of items assigned to the first partition (default = 0.5).

    Returns:
    - New dictionary 1 (dict).
    - New dictionary 2 (dict).
    """
    keys = list(input_dict.keys())

    # Calculate the number of keys for the first partition
    dict1_size = int(len(keys) * prop)

    # Randomly sample keys for the first partition
    random.seed(10) # to make sure all pricing mechanisms are compared based on the same bids partition
    dict1_keys = random.sample(keys, dict1_size)

    # Create the first partition
    dict1 = {key: input_dict[key] for key in dict1_keys}

    # Create the second partition with the remaining keys
    dict2 = {key: input_dict[key] for key in keys if key not in dict1_keys}

    return dict1, dict2



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
def max_epc_rev(value_cdf, lower, upper, **kwargs):
    """
    Maximizes the expected per capita revenue, i.e., max_p p(1 - F(p)).

    Parameters:
    - value_pdf (callable func): Cumulative distribution function of buyers' values.
    - lower (float): Lower limit for bidder values and bids.
    - upper (float): Upper limit for bidder values and bids.
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



# Scale the value
def scale_value(value, lower, old_upper, new_upper):
    """
    Scales the random value generated from [lower, old_upper] as if from [lower, new_upper].

    Parameters:
    - value (float): Value to scale.
    - lower (float): Lower limit of the range where the value is generated from. 
    - old_upper (float): Upper limit of the range where the value is generated from.
    - new_upper (float): Upper limit of the range where the value is pretended to be generated from.

    Return:
    - Scaled value (float).
    """
    return lower + (value - lower) / (old_upper - lower) * (new_upper - lower)



# Scale the cdf
def scale_cdf(cdf, lower, old_upper, new_upper):
    """
    Scales the cdf with support of [lower, old_upper] to [lower, new_upper].

    Parameters:
    - cdf (Callable): cdf to scale.
    - lower (float): Lower limit of the range where the value is generated from. 
    - old_upper (float): Upper limit of the range where the value is generated from.
    - new_upper (float): Upper limit of the range where the value is pretended to be generated from.

    Return:
    - Scaled cdf (Callable).
    """
    def scaled_cdf(x):
        return scale_value(cdf(x), lower, old_upper, new_upper)
    
    return scaled_cdf
