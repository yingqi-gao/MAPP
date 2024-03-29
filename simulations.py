import pickle
import multiprocessing
from sim_auction import simulate_loop


# Retrieve histories
with open("histories.pkl", "rb") as file:
        histories = pickle.load(file)



# Store useful strings for generating a file name.
file_name_strings = {"upper_status": ["fix", "float"],
                     "scale_of_bidders": ["small", "large"],
                     "type_of_dist": ["uniform", "normal", "exponential"],
                     "pricing_mechanism": ["RSOP", "DOP", "RSDE", "KDE"]}



if __name__ == "__main__":    
    # Number of processes to spawn
    num_processes = 48


    # Create a list to store references to the processes
    processes = []
    params = {"prop": 0.5}
    

    # Create and start the processes
    for i in range(2):
        upper_status = file_name_strings["upper_float"][i]
        if upper_status == "fix":
             params["upper_float"] = False
        elif upper_status == "float":
             params["upper_float"] = True

        for j in range(2):
            scale_of_bidders = file_name_strings["scale_of_bidders"][j]

            for k in range(3):
                type_of_dist = file_name_strings["type_of_dist"][k]
                history = histories[upper_status][scale_of_bidders][type_of_dist]
                params = {"history": history}

                for l in range(4):
                    pricing_mechanism = file_name_strings["pricing_mechanism"][l]
                    params["pricing"] = pricing_mechanism
                    file_name = ("_").join([upper_status, scale_of_bidders, type_of_dist, pricing_mechanism]) + ".pkl"
                    params["file_name"] = file_name
                    if pricing_mechanism == "RSDE":
                         params["repeated"] = True
                    else:
                         params["repeated"] = False

                    ## Start the actual simulations.
                    process = multiprocessing.Process(target = simulate_loop, kwargs = params)
                    processes.append(process) 
                
    for process in processes:
        process.start()

    # Wait for all processes to finish
    for process in processes:
        process.join()

    print("All processes have finished.")