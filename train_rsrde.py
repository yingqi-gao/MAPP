from initialize_auctions import AuctionInitialization
from _py_density_estimation import get_bw, rde_training_py
import dill
from multiprocessing import Lock, Process



def train_rsrde(lock,
                initializations: list[AuctionInitialization], 
                initializations_name: str):
    with lock:
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



def main():
    sub_lock = Lock()
    processes = [] # num_processes = 12

    import os
    os.environ['R_HOME'] = '/u/home/y/yqg36/.conda/envs/rpy2-env/lib/R'

    for filename in os.scandir("data/inits/"):
        if filename.path.endswith("0.pkl"):
            with open(filename.path, "rb") as file:
                initializations = dill.load(file)
            initializations_name = filename.path.split("/")[2].split(".")[1]
            process = Process(target = train_rsrde, 
                              kwargs = {"lock": sub_lock, 
                                        "initializations": initializations, 
                                        "initializations_name": initializations_name})
            processes.append(process) 
                    
    for process in processes:
        process.start()

    # Wait for all processes to finish
    for process in processes:
        process.join()

    print("~~~~~~~~~~~~~~~~~~~~")
    print("All done done!!!")
    


if __name__ == "__main__":
    main()
