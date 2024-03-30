# import os
import rpy2.robjects as robjects
# os.environ['R_HOME'] = '/u/home/y/yqg36/.conda/envs/rpy2-env/lib/R'
import functools


# import density estimation functions in r
robjects.r(
    '''
    source('_r_density_estimation.r')
    '''
)



# Convert between Python objects and R objects.
def py2r(obj):
    """
    Converts a Python (nested) numeric list to an R object.

    Parameter:
    - obj (any): A Python object.

    Return:
    - An R object.
    """
    #print("######")
    if isinstance(obj, list):
        vals = [py2r(item) for item in obj]
        #print("obj", obj)
        #print("vals", vals)
        #print("types", [type(v) for v in vals])
        if all(isinstance(val, float) or isinstance(val, int) for val in vals):
            #print("input", obj, "output", robjects.FloatVector(vals))
            return robjects.FloatVector(vals)
        elif all(isinstance(val, robjects.vectors.FloatVector) for val in vals):
            #print("hereeeeee")
            return robjects.r.list(*vals)
        else:
            raise ValueError([type(val) for val in vals])
    elif any(isinstance(obj, type_) for type_ in [float, int, bool, str]):
        return obj
    raise ValueError()


def r2py(obj):
    """
    Converts an R numeric vector or an R list of numeric vectors to a (nested) Python list.

    Parameter:
    - obj (any): An R object.

    Return:
    - A Python object.
    """
    # FloatVector, ListVector
    # print([type(i) for i in obj])
    if isinstance(obj, robjects.vectors.FloatVector) or isinstance(obj, robjects.vectors.IntVector):
        output = [float(item) for item in obj]
        if len(obj) == 1:
            return output[0]
        else:
            return output
    elif isinstance(obj, robjects.vectors.ListVector):
        return [r2py(item) for item in obj]
    else:
        raise ValueError(type(obj))
    

def r2py_func_wrapper(func):
    """
    Wrap the R function so that the returned value is a Python object.

    Parameter: 
    - func (Callable): An R function used in Python.

    Return:
    - A function that returns a Python object.
    """
    @functools.wraps(func)
    def func_wrapped(*args, **kwargs):
        outputs_r = func(*args, **kwargs)
        return r2py(outputs_r)
    return func_wrapped



# Calculate the bandwidth
def get_bw(obs_at_t):
    """
    Calculates bandwidth for kernel density estimation.

    Parameter:
    - obs_at_t (list[num]): All (future training) observations received at round t, i.e., observations for estimating future density.

    Return:
    - Bandwidth selected for kernel density estimation based on observations at round t.
    """
    # Step 1: Convert the Python list to an R object
    obs_at_t = py2r(obs_at_t)

    # Step 2: Run the bw.SJ function in R
    bw_at_t = robjects.r['bw.SJ'](obs_at_t)

    # Step 3: Convert the R object back to a Python list
    bw_at_t = r2py(bw_at_t)

    # Return
    return bw_at_t



# Kernel density estimation
def kde_py(observations, lower, upper):
    """
    Kernel density estimation.

    Parameters: 
    - observations (list[num]): Observations for estimating current density.
    - lower (float): Lower support of all densities.
    - upper (float): Upper support of all densities.
    
    Return:
    - The estimated cdf function.
    """
    observations = py2r(observations)
    cdf = robjects.r["kde_r"](observations, lower, upper)
    return r2py_func_wrapper(cdf)



# Repeated density estimation
def rde_py(test_obs_at_t, lower, upper, *, train_hist, train_bws, method = "MLE", grid_size = 1024):
    """
    Repeated density estimation.

    Parameters:
    - test_obs_at_t (num list): Test observations received at round t, i.e., observations for estimating current density.
    - lower (float): Lower support of all densities.
    - upper (float): Upper support of all densities.
    Keyword Arguments
    - train_hist (list of num lists): Training history, i.e., stored training observations. Each element is a numeric list storing training observations at round t.
    - train_bws (num list): Bandwidths selected at each round for kernel density estimation.
    - method (str): A string specifying the method to use for calculating the estimated parameters ("MLE", "MAP", "BLUP", default: "MLE").
    - grid_size (int): The number of grid points to generate for evaluating estimated density (default: 1024).
    """
    # Step 1: Convert all Python inputs to acceptible R inputs. 
    params = locals()
    params["method"] = "FPCA_" + method
    params = {key: py2r(value) for key, value in params.items()}

    # Step 2: Call and run the repeated density estimation in R. 
    cdf = robjects.r["rde_r"](**params)
    return r2py_func_wrapper(cdf)


if __name__ == "__main__":
    print("running tests...")

    # test for py2r
    print(py2r([1,2,3]))
    print(py2r(1))
    print(py2r(2.0))
    print(py2r(True))
    print(py2r("www"))
    print(py2r([1,2,3.4]))
    print(py2r([[1,2,3], [4,5,6]]))
    print(py2r([[1,2,3], [1,2,3,4.9]]))
    print(py2r([[1.0]]))
    print(py2r([[1,2,3], [True, "the"]]))
    print(py2r([[1, 2, 3], 2, 4.5]))
    print(r2py(py2r([1,2,3])) == [1,2,3])

    # test for r2py
    test_r = robjects.r('''function(x) x''')
    print(r2py(test_r(1)))
    print(r2py(test_r(1.0)))
    print(r2py(test_r(True)))
    print(r2py(test_r("string")))
    print(r2py(robjects.r('''list(c(1, 2, 3), TRUE, 5)''')))
    print(r2py(py2r([[1,2,3], [1,2,3,4.9]]))==[[1,2,3], [1,2,3,4.9]])
    print(r2py(py2r([[1.0]]))==[1.0])
