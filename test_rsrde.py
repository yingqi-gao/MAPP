import os
os.environ['R_HOME'] = '/u/home/y/yqg36/.conda/envs/rpy2-env/lib/R'
import pickle
import gc
import sys
import time
import numpy as np
from _pricing_mechanisms import RSRDE



def run_rsrde(dist_type: str,
              repetition_no: str,
              num_training_bids: str,
              num_testing_bids: int):
    history_training_results_file = "data/sim/" + dist_type + "/train_results_" + dist_type + "_rep" + repetition_no + "_" + num_training_bids + "bids.pkl"
    with open(history_training_results_file, "rb") as file:
        history_training_results = pickle.load(file)
    
    testing_info_file = "data/sim/" + dist_type + "/test_info_" + dist_type + ".pkl"
    with open(testing_info_file, "rb") as file:
        testing_info = pickle.load(file)
    testing_true_dist = testing_info["true_dist"]
    
    testing_bids = testing_info["testing_bids"][:num_testing_bids]
    testing_results = []
    for t in range(20, 100):
        price, _ = RSRDE(testing_bids, lower = 1, upper = 10, random_seed = 666, training_results = history_training_results)
        revenue = testing_true_dist.get_actual_revenue(price)
        testing_results.append((price, revenue))

    with open("data/sim/" + dist_type + "/test_results_" + dist_type + "_rep" + repetition_no + "_" + num_training_bids + "training_" + str(num_testing_bids) + "testing.pkl", "wb") as file:
        pickle.dump(testing_results, file)
    print(f"Done with round {t + 1} of {dist_type} with {num_training_bids} bids per round for {str(num_testing_bids)} testing bids - Repetition No.{repetition_no}!")

    print("--------------------")
    print(f"All done with testing on {dist_type} with {num_training_bids} bids per round for {str(num_testing_bids)} testing bids - Repetition No.{repetition_no}!")



def main():
    t_start = time.time()
    dist_type = sys.argv[1]
    repetition_no = sys.argv[2]
    num_training_bids = sys.argv[3]
    num_testing_bids = int(sys.argv[4])
    run_rsrde(dist_type = dist_type, 
              repetition_no = repetition_no, 
              num_training_bids = num_training_bids, 
              num_testing_bids = num_testing_bids)
    print(f"It took {time.time() - t_start} seconds.")

if __name__ == "__main__":
    main()