from _classes_auction import Auction, DOPAuction, RSOPAuction, RSKDEAuction, RSRDEAuction
import dill
import gc
import sys



def run_auctions(initializations_name: str) -> list[Auction]:
    with open("data/inits/" + initializations_name + ".pkl", "rb") as file:
        initializations = dill.load(file)
    with open("data/RSRDE_training/" + initializations_name + "_RSRDE_training.pkl", "rb") as file:
        training_results = dill.load(file)

    num_rounds = len(initializations)
    DOP_auctions = []    
    for i in range(num_rounds):
        init = initializations[i]
        training_history = training_results[i]

        DOP_auction = DOPAuction(initialization = init)
        DOP_auctions.append(DOP_auction)

        RSOP_auctions, RSKDE_auctions, RSRDE_auctions = [], [], []
        for j in range(200):
            RSOP_auctions.append(RSOPAuction(initialization = init, random_seed = j))
            RSKDE_auctions.append(RSKDEAuction(initialization = init, random_seed = j))
            if training_history is not None:
                RSRDE_auctions.append(RSRDEAuction(initialization = init, random_seed = j, training_history = training_history))

        with open("data/Rep200/" + initializations_name + "_DOP.pkl", "wb") as file:
            dill.dump(DOP_auctions, file)
        if i == 0:
            with open("data/Rep200/" + initializations_name + "_RSOP.pkl", "wb") as file:
                dill.dump(RSOP_auctions, file)
            with open("data/Rep200/" + initializations_name + "_RSKDE.pkl", "wb") as file:
                dill.dump(RSKDE_auctions, file)
        else:
            with open("data/Rep200/" + initializations_name + "_RSOP.pkl", "ab") as file:
                dill.dump(RSOP_auctions, file)
            with open("data/Rep200/" + initializations_name + "_RSKDE.pkl", "ab") as file:
                dill.dump(RSKDE_auctions, file)
        if i == 20:
            with open("data/Rep200/" + initializations_name + "_RSRDE.pkl", "wb") as file:
                dill.dump(RSRDE_auctions, file)
        elif i > 20:
            with open("data/Rep200/" + initializations_name + "_RSRDE.pkl", "ab") as file:
                dill.dump(RSRDE_auctions, file)
        
        print(f"Round {i + 1} of {initializations_name} done!")
        gc.collect()
            
    print("--------------------")
    print(f"All done with {initializations_name}!")


arg = sys.argv[1]
run_auctions(arg)