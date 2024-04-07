from _classes_initialization import OnlineAuctionRandomInitialization
import rpy2.robjects as robjects
from _classes_auction import Auction, DOPAuction, RSOPAuction, RSKDEAuction, RSRDEAuction
import dill
from multiprocessing import Lock, Process
from _run_auctions import run_auctions
from _classes_initialization import OnlineAuctionRandomInitialization



def run_auctions(lock,
                 online_initialization: OnlineAuctionRandomInitialization, 
                 online_initialization_name: str,
                 pricing_mechanism: float,
                 training_results: list[robjects.vectors.ListVector] = None) -> list[Auction]:
    # acquire the lock
    with lock:
        num_rounds = online_initialization.num_rounds

        sequence_auctions = []
        for i in range(num_rounds):
            auction_initialization = online_initialization.sequence_auctions[i]

            if pricing_mechanism == "DOP":
                auction = DOPAuction(initialization = auction_initialization)
                sequence_auctions.append(auction)

            else:
                repetition_auctions = []
                for random_seed in range(500):
                    if pricing_mechanism == "RSOP":
                        auction = RSOPAuction(initialization = auction_initialization)
                    elif pricing_mechanism == "RSKDE":
                        auction = RSKDEAuction(initialization = auction_initialization,
                                               random_seed = random_seed)
                    elif pricing_mechanism == "RSRDE": 
                        training_history = training_results[i]
                        if training_history is None:
                            auction = None
                        else:
                            auction = RSRDEAuction(initialization = auction_initialization, 
                                                   random_seed = random_seed,
                                                   training_history = training_history)
                repetition_auctions.append(auction)        
                sequence_auctions.append(repetition_auctions)

            if i % 10 == 9:
                file_name = "data/" + online_initialization_name + "_" + pricing_mechanism + "_rep500.pkl"
                with open(file_name, "wb") as file:
                    dill.dump(sequence_auctions, file)
                print(f"Round {i + 1} of {file_name} done!")
            
        print("--------------------")
        print(f"All done with {pricing_mechanism} on {online_initialization_name}!")



def run_process(online_initialization: OnlineAuctionRandomInitialization, 
                online_initialization_name: str,
                training_results: list[robjects.vectors.ListVector]):
    pricing_mechanism_reference = ["RSOP", "RSKDE", "RSRDE"] # RSRDE_MLE

    sub_lock = Lock()
    processes = [] # num_processes = 3

    for pricing_mechanism in pricing_mechanism_reference:
        params = {
            "lock": sub_lock,
            "online_initialization": online_initialization,
            "online_initialization_name": online_initialization_name,
            "pricing_mechanism": pricing_mechanism,
            "training_results": training_results
        }

        process = Process(target = run_auctions, kwargs = params)
        processes.append(process)

    for process in processes:
        process.start()
    
    for process in processes:
        process.join()
    
    print("====================")
    print(f"All done with {online_initialization_name}!")



def main():
    with open("data/initializations.pkl", "rb") as file:
        online_auction_initializations = dill.load(file)

    processes = [] # num_processes = 6, 3 mechanisms applied to each initialization, 500 repetitions per mechanism 

    for name, initialization in online_auction_initializations.items():
        if name.split("_")[1] == "fixed":
            training_history_name = "data/RSRDE_training/" + name + "_RSRDE_training.pkl"
            with open(training_history_name, "rb") as file:
                training_results = dill.load(file)

            process = Process(target = run_process, 
                              kwargs = {"online_initialization": initialization, 
                                        "online_initialization_name": name,
                                        "training_results": training_results})
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
