import os
import dill
import numpy as np
import matplotlib.pyplot as plt



def get_regrets(initialization_name: str, num_rounds = 200):
    
    regrets_to_plot = {}
    
    for file_name in os.listdir("data"):
        if file_name.startswith(initialization_name):
            pricing_mechanism = file_name.split("_")[3]
            if pricing_mechanism == "RSRDE":
                method = file_name.split("_")[4].split(".")[0]
                pricing_mechanism = pricing_mechanism + "_" + method
            
            with open("data/" + file_name, "rb") as file:
                online_auctions = dill.load(file)
                
            if len(online_auctions) is not num_rounds:
                raise ValueError(f"Number of rounds is {len(online_auctions)}, supposed to be {num_rounds}!")
            
            regret = [auction.regret for auction in online_auctions if auction is not None]
            
            regrets_to_plot[pricing_mechanism] = regret
    
    return regrets_to_plot
    


def main():
    with open("data/initializations.pkl", "rb") as file:
        initializations = dill.load(file)

    for name in initializations.keys():
        regrets_to_plot = get_regrets(name)
    
        plt.figure()
    
        for pricing_mechanism, regrets in regrets_to_plot.items():
             x = np.arange(0, 200) # number of rounds = 200
             if pricing_mechanism.startswith("RSRDE"):
                 x = np.arange(101, 200)
             plt.plot(x, regrets, label = pricing_mechanism, linewidth = 0.5)
                
        plt.xlabel('round')
        plt.ylabel('regret')
        plt.legend()
        plt.title(name)
    
        plt.savefig(name + ".png")

        

if __name__ == "__main__":
        main()
