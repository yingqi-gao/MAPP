import os
import rpy2.robjects as robjects


os.environ['R_HOME'] = '/u/home/y/yqg36/.conda/envs/rpy2-env/lib/R'
robjects.r(
    '''
    source('r_utils.r')
    ''')


# Convert between Python objects and R objects
def py2rlist(pylist):
    """
    Converts a Python list to an R vector. If the Python list is nested, convert to an R list of vectors.

    Parameter:
    - pylist (list): A Python list of numerical values or of lists of numerical values.

    Return:
    - An R vector or an R list of vectors.
    """
    # Change the input only if it is a list.
    if isinstance(pylist, list):
        # If any element of the input list is again a list, convert the outer list to an R list.
        if any([isinstance(e, list) for e in pylist]):
            result = robjects.r.list(*[py2rlist(e) for e in pylist])
        # If no element of the input list is a list, convert the whole list into an R float vector.
        else:
            result = robjects.FloatVector(pylist)

    # If the input is not a list, do nothing.
    else:
        result = pylist

    # Return
    return result

def r2pylist(rinput):
    """
    Converts an R vector or an R list of vectors to a (nested) Python list.

    Parameter:
    - rinput (R list or R vec): An R list of vectors or an R vector.

    Return:
    - A Python list.
    """
    # If the input is an R list of vectors, convert to a nested Python list.
    if isinstance(rinput, robjects.vectors.ListVector):
        pylist = [r2pylist(e) for e in rinput]

    # If the input is not an R list, convert it to a Python list.
    elif isinstance(rinput, robjects.vectors.FloatVector) or isinstance(rinput, robjects.vectors.IntVector):
        pylist = [float(e) for e in rinput]
        if len(pylist) == 1:
            pylist = pylist[0]

    else:
        pylist = rinput

    # Return
    return pylist


# Calculate the bandwidth
def get_bw(obs_at_t):
    """
    Calculates bandwidth for kernel density estimation.

    Parameter:
    - obs_at_t (num *list*): All (future training) observations received at round t, i.e., observations for estimating future density.

    Return:
    - Bandwidth selected for kernel density estimation based on observations at round t.
    """
    # Step 1: Convert the Python list to an R object
    obs_at_t = py2rlist(obs_at_t)

    # Step 2: Run the bw.SJ function in R
    bw_at_t = robjects.r['bw.SJ'](obs_at_t)

    # Step 3: Convert the R object back to a Python list
    bw_at_t = r2pylist(bw_at_t)

    # Return
    return bw_at_t
print(get_bw([1, 2, 3, 4, 5]))

# Estimate density
def density_est(train_hist, train_bws, test_obs_at_t, lower, upper, grid_size = 1024, method = "MLE", repeated = True):
    '''
    Estimates density at round t.

    Parameters:
    - train_hist (list of num *lists*): Training history, i.e., stored training observations. Each element is a numeric *list* storing training observations at round t.
    - train_bws (num *list*): Bandwidths selected at each round for kernel density estimation.
    - test_obs_at_t (num *list*): Test observations received at round t, i.e., observations for estimating current density.
    - lower (*float*): Lower support of all densities.
    - upper (*float*): Upper support of all densities.
    - grid_size (int): The number of grid points to generate for evaluating estimated density.
    - method (*str*): A string specifying the method to use for calculating the estimated parameters (*"MLE", "MAP", "BLUP"*).
    - repeated (bool): Whether to use historcial data for density estimation (default: True). 

    Return:
    - The estimated cdf function.
    
    Side Effect:
    - Stores and flags estimated parameters if any.
    '''
    # Step 1: Convert all inputs to R objects.
    params = locals()
    del params["repeated"]
    params = dict((key, py2rlist(value)) for key, value in params.items())


    # Step 2: Convert the method parameter to the correct string.
    params["method"] = "FPCA_" + params["method"]


    # Step 3: Run the density estimation function in R.
    if repeated:
        ## Use the method in “Nonparametric Estimation of Repeated Densities with Heterogeneous Sample Sizes” for repeated density estimation.
        r_results = robjects.r["rep_den_est_r"](**params)
        est_params, est_cdf_r = r2pylist(r_results)
        global estimated_parameters_stored, estimated_parameters_history
        if not estimated_parameters_stored: 
            estimated_parameters_stored = True
        estimated_parameters_history.append(est_params)
    else:
        ## Use the kernel density estimation method with no historical data. 
        est_cdf_r = robjects.r["kde_r"](params["test_obs_at_t"], params["lower"], params["upper"])


    # Step 4: Convert R outputs to Python objects
    def est_cdf(x):
        x_r = robjects.FloatVector(x)
        result = est_cdf_r(x_r)
        return float(result[0])

    # Return
    return est_cdf