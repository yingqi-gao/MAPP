import dill
import multiprocessing
from _run_auctions import run_auctions
from _classes_initialization import OnlineAuctionRandomInitialization



def run_process(online_initialization: OnlineAuctionRandomInitialization, 
                online_initialization_name: str):
    pricing_mechanism_reference = ["DOP", "RSOP", "RSKDE", "RSRDE_MLE", "RSRDE_MAP", "RSRDE_BLUP"]

    processes = [] # num_processes = 6
    for pricing_mechanism in pricing_mechanism_reference:
        params = {
            "online_initialization": online_initialization,
            "online_initialization_name": online_initialization_name,
            "pricing_mechanism": pricing_mechanism
        }

        process = multiprocessing.Process(target = run_auctions, kwargs = params)
        processes.append(process)

        print("###############################")
        print(f"All done with {pricing_mechanism} on {online_initialization_name}!")
    
    for process in processes:
        process.start()
    
    for process in processes:
        process.join()
    
    print("###############################")
    print("###############################")
    print(f"All done with {online_initialization_name}!")



# def main():
#     with open("initializations.pkl", "rb") as file:
#         online_auction_initializations = pickle.load(file)

#     num_processes = 12
#     processes = []

#     for name, initialization in online_auction_initializations.items():
#         process = multiprocessing.Process(target = run_process, kwargs = {"name": name, "initialization": initialization})
#         processes.append(process) 
                    
#     for process in processes:
#         process.start()

#     # Wait for all processes to finish
#     for process in processes:
#         process.join()

#     print("All processes have finished.")
        
def main():
    with open("initializations.pkl", "rb") as file:
        online_auction_initializations = dill.load(file)

    test_flag = True
    for name, initialization in online_auction_initializations.items():
        run_process(initialization, name)
        if test_flag:
            break

    print("All done done!!!")    


if __name__ == "__main__":
    main()
