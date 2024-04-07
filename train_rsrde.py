from _classes_initialization import OnlineAuctionRandomInitialization
from _py_density_estimation import get_bw, rde_training_py
import dill
from multiprocessing import Lock, Process



def train_rsrde(lock,
                online_initialization: OnlineAuctionRandomInitialization, 
                online_initialization_name: str):
    with lock:
        num_rounds = online_initialization.num_rounds
        lower = online_initialization.lower
        upper = online_initialization.upper

        history_training_data = []
        history_training_bandwidths = []
        history_training_results = []

        for i in range(num_rounds):
            auction_initialization = online_initialization.sequence_auctions[i]
            if auction_initialization.true_dist.lower != lower or auction_initialization.true_dist.upper != upper:
                raise ValueError("The assumption of a common support is violated!")

            if i < 10:
                training_results = None
            else:
                training_results = rde_training_py(train_hist = history_training_data,
                                                   train_bws = history_training_bandwidths,
                                                   lower = lower,
                                                   upper = upper)
            history_training_results.append(training_results)
                
            bids_list = [*auction_initialization.bids.values()]
            bandwidth = get_bw(bids_list)
            history_training_data.append(bids_list)
            history_training_bandwidths.append(bandwidth)
        
            if i % 10 == 9:
                file_name = "data/RSRDE_training/" + online_initialization_name + "_RSRDE_training.pkl"
                with open(file_name, "wb") as file:
                    dill.dump(history_training_results, file)
                print(f"Round {i + 1} of {file_name} done!")

    print("--------------------")
    print(f"All done with RSRDE training on {online_initialization_name}!")



def main():
    with open("data/initializations.pkl", "rb") as file:
        online_auction_initializations = dill.load(file)

    sub_lock = Lock()
    processes = [] # num_processes = 6

    for name, initialization in online_auction_initializations.items():
        if name.split("_")[1] == "fixed":
            process = Process(target = train_rsrde, 
                              kwargs = {"lock": sub_lock, 
                                        "online_initialization": initialization, 
                                        "online_initialization_name": name})
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
