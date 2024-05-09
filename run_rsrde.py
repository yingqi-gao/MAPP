import os
os.environ['R_HOME'] = '/u/home/y/yqg36/.conda/envs/rpy2-env/lib/R'
import sys
from initialize_auctions import get_training_bids
from train_rsrde import train_rsrde
from test_rsrde import test_rsrde



def run_rsrde(dist_type: str,
              repetition_no: str):
    get_training_bids(dist_type = dist_type,
                      repetition_no = repetition_no)
    
    for num_training_bids in [50, 100, 500]:
        train_rsrde(dist_type = dist_type,
                    repetition_no = repetition_no,
                    num_training_bids = num_training_bids)
        
        for num_testing_bids in [10, 50, 100, 500]:
            test_rsrde(dist_type = dist_type,
                       repetition_no = repetition_no,
                       num_training_bids = str(num_training_bids),
                       num_testing_bids = num_testing_bids)
            
    print(f"All done with {dist_type} - Repetition No.{repetition_no}!")



def main():
    dist_type = sys.argv[1]
    repetition_no = sys.argv[2]
    run_rsrde(dist_type = dist_type, repetition_no = repetition_no)