import dill
from multiprocessing import Lock, Process
from _run_auctions import run_auctions
from _classes_initialization import OnlineAuctionRandomInitialization



def run_process(online_initialization: OnlineAuctionRandomInitialization, 
                online_initialization_name: str):
    pricing_mechanism_reference = ["DOP", "RSOP", "RSKDE", "RSRDE_MLE"]

    sub_lock = Lock()
    processes = [] # num_processes = 4

    for pricing_mechanism in pricing_mechanism_reference:
        params = {
            "lock": sub_lock,
            "online_initialization": online_initialization,
            "online_initialization_name": online_initialization_name,
            "pricing_mechanism": pricing_mechanism
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

    processes = [] # num_processes = 12, each with 6 sub-processes

    for name, initialization in online_auction_initializations.items():
        process = Process(target = run_process, kwargs = {"online_initialization": initialization, "online_initialization_name": name})
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
