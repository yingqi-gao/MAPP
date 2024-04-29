import os
os.environ['R_HOME'] = '/u/home/y/yqg36/.conda/envs/rpy2-env/lib/R'
import dill
import sys
from _py_density_estimation import get_bw, rde_training_py



def train_rsrde(initializations_name: str):
    with open("data/inits/" + initializations_name + ".pkl", "rb") as file:
        initializations = dill.load(file)
    num_rounds = len(initializations)
    lower = initializations[0].true_dist.lower
    upper = initializations[0].true_dist.upper

    history_training_data = []
    history_training_bandwidths = []
    history_training_results = []

    for i in range(num_rounds):
        init = initializations[i]
        if init.true_dist.lower != lower or init.true_dist.upper != upper:
            raise ValueError("The assumption of a common support is violated!")

        if i < 20:
            training_results = None
        else:
            training_results = rde_training_py(train_hist = history_training_data,
                                                train_bws = history_training_bandwidths,
                                                lower = lower,
                                                upper = upper)
        history_training_results.append(training_results)
            
        bids = [*init.bids.values()]
        bandwidth = get_bw(bids)
        history_training_data.append(bids)
        history_training_bandwidths.append(bandwidth)
    
        if i % 10 == 9:
            file_name = "data/RSRDE_training/" + initializations_name + "_RSRDE_training.pkl"
            with open(file_name, "wb") as file:
                dill.dump(history_training_results, file)
            print(f"Round {i + 1} of {file_name} done!")

    print("--------------------")
    print(f"All done with RSRDE training on {initializations_name}!")


arg = sys.argv[1]
train_rsrde(arg)
