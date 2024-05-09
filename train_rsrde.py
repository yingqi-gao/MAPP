import os
os.environ['R_HOME'] = '/u/home/y/yqg36/.conda/envs/rpy2-env/lib/R'
import pickle
import sys
from _py_density_estimation import get_bw, rde_training_py



def train_rsrde(dist_type: str,
                repetition_no: str,
                num_training_bids: int):
    with open("data/sim/" + dist_type + "/train_bids_" + dist_type + "_rep" + repetition_no + ".pkl", "rb") as file:
        training_bids = pickle.load(file)

    history_training_data = []
    history_training_bandwidths = []
    history_training_results = []

    file_name = "data/sim/" + dist_type + "/train_results_" + dist_type + "_rep" + repetition_no + "_" + str(num_training_bids) + "bids.pkl"
    for t in range(100):
        if t < 20:
            pass
        else:
            training_results = rde_training_py(train_hist = history_training_data,
                                               train_bws = history_training_bandwidths,
                                               lower = 1, upper = 10)
            history_training_results.append(training_results)
        
        training_bids_at_t = [*training_bids[t].values()]
        bids = training_bids_at_t[:num_training_bids]
        bandwidth = get_bw(bids)
        history_training_data.append(bids)
        history_training_bandwidths.append(bandwidth)

        with open(file_name, "wb") as file:
            pickle.dump(history_training_results, file)
        print(f"Done with round {t + 1} of {dist_type} with {num_training_bids} bids per round - Repetition No.{repetition_no}!")

    print("--------------------")
    print(f"All done with RSRDE training on {dist_type} with {num_training_bids} bids per round - Repetition No.{repetition_no}!")



def main():
    dist_type = sys.argv[1]
    repetition_no = sys.argv[2]
    num_training_bids = int(sys.argv[3])
    train_rsrde(dist_type = dist_type, repetition_no = repetition_no, num_training_bids = num_training_bids)    
    
if __name__ == "__main__":
    main()