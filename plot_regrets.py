import dill
from _classes_auction import OnlineAuctions
import numpy as np
from dataclasses import fields
import math
import matplotlib.pyplot as plt



def plot_regret(online_auctions_name: str):
        with open(online_auctions_name + ".pkl", "rb") as file:
                online_auctions = dill.load(file)
        print(len(online_auctions.DOP_results))

        # num_rounds = online_auctions.online_initialization.num_rounds
        # x = np.arange(0, num_rounds)

        # fig, ax = plt.subplots()
        # for auctions_result in fields(online_auctions):
        #         if auctions_result.name.endswith("results"):
        #                 pricing_mechanism = auctions_result.name.split("_")[0]
        #                 regret = [per_auction_results.regret for per_auction_results in getattr(online_auctions, auctions_result.name)]

        #                 if auctions_result.name.split("_")[0] == "RSRDE":
        #                         method = auctions_result.name.split("_")[1]
        #                         x = x[math.floor(num_rounds / 2):]
        #                         pricing_mechanism = pricing_mechanism + method

        #                 ax.plot(x, regret, label = pricing_mechanism)

        # legend = ax.legend()
        # plt.xlabel('round')
        # plt.ylabel('regret')
        # plt.show()
        # plt.savefig(online_auctions_name + ".png")



def main():
        with open("initializations.pkl", "rb") as file:
                online_auction_initializations = dill.load(file)
        
        for name in online_auction_initializations.keys():
                plot_regret(name)
        
        plot_regret("uniform_fixed_small")
        

if __name__ == "__main__":
        main()
