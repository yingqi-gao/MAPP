import pickle
import multiprocessing
from _classes_auction_result import OnlineAuctions



def run_process(name, initialization):
    online_auctions = OnlineAuctions(online_initialization = initialization)
    with open(name + ".pkl", "wb") as file:
        pickle.dump(online_auctions, file)



def main():
    with open("initializations.pkl", "rb") as file:
        online_auction_initializations = pickle.load(file)

    num_processes = 12
    processes = []

    for name, initialization in online_auction_initializations.items():
        process = multiprocessing.Process(target = run_process, kwargs = {"name": name, "initialization": initialization})
        processes.append(process) 
                    
    for process in processes:
        process.start()

    # Wait for all processes to finish
    for process in processes:
        process.join()

    print("All processes have finished.")
    


if __name__ == "__main__":
    main()
