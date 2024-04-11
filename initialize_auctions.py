from dataclasses import dataclass, field
from _classes_true_distribution import TrueDistribution, UniformDistribution, NormalDistribution, ExponentialDistribution
import random
import dill



@dataclass
class AuctionInitialization:
     true_dist: TrueDistribution
     num_bidders: int
     bids: dict[str, float] = field(init = False)

     def __post_init__(self):
        self.bids = self.true_dist.generate_bids(self.num_bidders)
        if len(self.bids) != self.num_bidders:
            raise ValueError("Number of bids per round not correct!")



def get_initializations(dist_type: str, num_bidders: int, num_rounds = 200, lower = 1, upper = 10):
    initializations = []
    true_dist_reference = {"uniform": UniformDistribution,
                           "normal": NormalDistribution,
                           "exponential": ExponentialDistribution}
    
    for i in range(num_rounds):
        random.seed(i)
        init = AuctionInitialization(true_dist = true_dist_reference[dist_type](lower = lower, 
                                                                                upper = upper),
                                     num_bidders = num_bidders)
        initializations.append(init)

    if len(initializations) != num_rounds:
        raise ValueError("Repetition of auction initializations not correct!")

    return initializations



def main():
    for dist_type in ["uniform", "normal", "exponential"]:
        for num_bidders in [10, 100, 1000, 10000]:
            initializations = get_initializations(dist_type, num_bidders)
            file_name = "_".join([dist_type, str(num_bidders)])
            with open("data/inits/" + file_name + ".pkl", "wb") as file:
                dill.dump(initializations, file)
            print(f"All done with {file_name}!")



if __name__ == "__main__":
    main()