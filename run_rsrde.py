import os
os.environ['R_HOME'] = '/u/home/y/yqg36/.conda/envs/rpy2-env/lib/R'
from concurrent.futures import ProcessPoolExecutor
import sys
import pickle
from initialize_auctions import get_training_bids
from train_rsrde import train_rsrde
from test_rsrde import test_rsrde



def run_rsrde(dist_type: str,
              repetition_no: str,
              num_training_bids: str,
              num_testing_bids: str):
    get_training_bids(dist_type = dist_type,
                      repetition_no = repetition_no)
    
    train_rsrde(dist_type = dist_type,
                repetition_no = repetition_no,
                num_training_bids = int(num_training_bids))
        
    test_rsrde(dist_type = dist_type,
               repetition_no = repetition_no,
               num_training_bids = num_training_bids,
               num_testing_bids = int(num_testing_bids))
            
    print(f"All done with {dist_type} - Repetition No.{repetition_no}: {num_training_bids} training bids and {num_testing_bids} testing bids!")


def check_length(full_path: str):
    with open(full_path, "rb") as file:
        testing_results = pickle.load(file)
    return len(testing_results)


def main():
    dist_type = sys.argv[1]

    folder_path = "data/sim/" + dist_type 
    test_prefix = "test_results"

    # Get all files in the folder
    files = os.listdir(folder_path)

    # Filter files that start with the specified prefix
    test_files = [file for file in files if file.startswith(test_prefix)]
    
    to_runs = []
    for i in range(100):
        for num_training_bids in ["50", "100", "500"]:
            for num_testing_bids in ["10", "50", "100", "500"]:
                test_file = "test_results_" + dist_type + "_rep" + str(i+1) + "_" + num_training_bids + "training_" + str(num_testing_bids) + "testing.pkl"
                if (test_file not in test_files) or (check_length(folder_path + "/" +test_file) != 80):
                    to_runs.append((i, num_training_bids, num_testing_bids))
    
    futures = []
    with ProcessPoolExecutor(max_workers=30) as executor:
        for i, num_training_bids, num_testing_bids in to_runs:
            futures.append(executor.submit(run_rsrde, 
                                           dist_type=dist_type, 
                                           repetition_no=str(i+1),
                                           num_training_bids=num_training_bids,
                                           num_testing_bids=num_testing_bids))
    for future in futures: 
        future.result()

if __name__ == "__main__":
    main()