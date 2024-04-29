from dataclasses import dataclass, field
from _classes_true_distribution import TrueDistribution, UniformDistribution, NormalDistribution, ExponentialDistribution, ParetoDistribution, LognormalDistribution
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



def get_initializations(dist_type: str, 
                        num_bidders_train: int, 
                        num_bidders_test: int, 
                        num_rounds_train: int,
                        num_rounds = 200, 
                        lower = 1, 
                        upper = 10):
    initializations = []
    true_dist_reference = {"uniform": UniformDistribution,
                           "normal": NormalDistribution,
                           "exponential": ExponentialDistribution,
                           "pareto": ParetoDistribution,
                           "lognormal": LognormalDistribution}
    
    for i in range(num_rounds):
        random.seed(i)
        if i < num_rounds_train:
            num_bidders = num_bidders_train
        else:
            num_bidders = num_bidders_test
        init = AuctionInitialization(true_dist = true_dist_reference[dist_type](lower = lower, 
                                                                                upper = upper),
                                     num_bidders = num_bidders)
        initializations.append(init)

    if len(initializations) != num_rounds:
        raise ValueError("Repetition of auction initializations not correct!")

    return initializations



def main():
    for dist_type in ["uniform", "normal", "exponential", "pareto", "lognormal"]:
        for num_bidders_train in [100, 200]:
            for num_rounds_train in [20, 50]:
                initializations = get_initializations(dist_type = dist_type, 
                                                      num_bidders_train = num_bidders_train,
                                                      num_bidders_test = 10,
                                                      num_rounds_train = num_rounds_train)
                file_name = "_".join([dist_type, str(num_bidders_train), str(num_rounds_train)])
                with open("data/inits/" + file_name + ".pkl", "wb") as file:
                    dill.dump(initializations, file)
                print(f"All done with {file_name}!")



if __name__ == "__main__":
    main()