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
            else:
                pricing_mechanism = pricing_mechanism.split(".")[0]
            
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
    
    fig, axs = plt.subplots(6, sharex=True, sharey=True)
    fig.suptitle(name)
    
    for i, (pricing_mechanism, regrets) in enumerate(regrets_to_plot.items()):
        x = np.arange(0, 200) # number of rounds = 200
        if pricing_mechanism.startswith("RSRDE"):
            x = np.arange(100, 200)
        axs[i].plot(x, regrets)
        axs[i].set_title(pricing_mechanism)
    
    fig.savefig("plots/" + "grid_" + name + ".png")
    
    # fig = plt.figure()
    # gs = fig.add_gridspec(2, 3, hspace=0, wspace=0)
    # fig, axs = plt.subplots(2, 3, sharex = True, sharey = True)
    
    # for i, (pricing_mechanism, regrets) in enumerate(regrets_to_plot.items()):
    #     x = np.arange(0, 200) # number of rounds = 200
    #     if pricing_mechanism.startswith("RSRDE"):
    #         x = np.arange(100, 200)
    #     # plt.plot(x, regrets, label = pricing_mechanism, linewidth = 0.5)
    #     axs[i].plot(x, regrets)
    #     axs[i].set_title(pricing_mechanism)
        
    # for ax in axs.flat:
    #     ax.set(xlabel='round', ylabel='regret')
             
    # for ax in axs.flat:
    #     ax.label_outer() 
                
    #     # plt.xlabel('round')
    #     # plt.ylabel('regret')
    #     # plt.legend()
    #     # plt.title(name)
    # fig.suptitle(name)
    
    # plt.savefig("plots/" + "grid_" + name + ".png")

       

if __name__ == "__main__":
        main()
