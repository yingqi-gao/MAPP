import os
os.environ['R_HOME'] = '/u/home/y/yqg36/.conda/envs/rpy2-env/lib/R'
from _classes_auction import Auction, DOPAuction, RSOPAuction, RSKDEAuction, RSRDEAuction
import dill
import gc
import sys
import warnings
from rpy2.rinterface import RRuntimeWarning
warnings.filterwarnings("ignore", category=RRuntimeWarning)



def run_auctions(initializations_name: str, pricing_mechanism: str) -> list[Auction]:
    with open("data/inits/" + initializations_name + ".pkl", "rb") as file:
        initializations = dill.load(file)
    num_rounds = len(initializations)

    if pricing_mechanism == "DOP":
        DOP_auctions = []
        for i in range(num_rounds):
            init = initializations[i]
            DOP_auction = DOPAuction(initialization = init)
            DOP_auctions.append(DOP_auction)
        with open("data/Rep200/" + initializations_name + "_DOP.pkl", "wb") as file:
            dill.dump(DOP_auctions, file)
        
    elif pricing_mechanism == "RSOP":
        for i in range(num_rounds):
            init = initializations[i]
            RSOP_auctions = []
            for j in range(200):
                RSOP_auctions.append(RSOPAuction(initialization = init, random_seed = j))
            if i == 0:
                with open("data/Rep200/" + initializations_name + "_RSOP.pkl", "wb") as file:
                    dill.dump(RSOP_auctions, file)
            else:
                with open("data/Rep200/" + initializations_name + "_RSOP.pkl", "ab") as file:
                    dill.dump(RSOP_auctions, file)
            print(f"Round {i + 1} of {pricing_mechanism} on {initializations_name} done!")

    elif pricing_mechanism == "RSKDE":
        for i in range(num_rounds):
            init = initializations[i]
            RSKDE_auctions = []
            for j in range(200):
                RSKDE_auctions.append(RSKDEAuction(initialization = init, random_seed = j))
            if i == 0:
                with open("data/Rep200/" + initializations_name + "_RSKDE.pkl", "wb") as file:
                    dill.dump(RSKDE_auctions, file)
            else:
                with open("data/Rep200/" + initializations_name + "_RSKDE.pkl", "ab") as file:
                    dill.dump(RSKDE_auctions, file)
            print(f"Round {i + 1} of {pricing_mechanism} on {initializations_name} done!")

    elif pricing_mechanism == "RSRDE":
        with open("data/RSRDE_training/" + initializations_name + "_RSRDE_training.pkl", "rb") as file:
            training_results = dill.load(file)
        for i in range(num_rounds):
            init = initializations[i]
            training_history = training_results[i]
            if training_history is None:
                RSRDE_auctions = None
            else:
                RSRDE_auctions = []
                for j in range(200):
                    RSRDE_auctions.append(RSRDEAuction(initialization = init, random_seed = j, training_history = training_history))
            if i == 0:
                with open("data/Rep200/" + initializations_name + "_RSRDE.pkl", "wb") as file:
                    dill.dump(RSRDE_auctions, file)
            else:
                with open("data/Rep200/" + initializations_name + "_RSRDE.pkl", "ab") as file:
                    dill.dump(RSRDE_auctions, file)
            print(f"Round {i + 1} of {pricing_mechanism} on {initializations_name} done!")

    else:
        raise ValueError("Pricing mechanism not recognized!") 
    
    print(f"All done with {pricing_mechanism} on {initializations_name}!")
    gc.collect()



arg1 = sys.argv[1]
arg2 = sys.argv[2]
run_auctions(arg1, arg2)