from dataclasses import dataclass, field, InitVar
from _classes_initialization import AuctionInitialization
from typing import Optional, Callable
from _pricing_mechanisms import DOP, RSOP, RSKDE, RSRDE
import rpy2.robjects as robjects



@dataclass
class Auction:
    initialization: AuctionInitialization
    pricing_mechanism: str
    actual_price: float
    actual_revenue: float = field(init = False)
    ideal_price: float = field(init = False)
    ideal_revenue: float = field(init = False)
    regret: float = field(init = False)
    training_history: InitVar[Optional[robjects.vectors.ListVector]] = None

    def __post_init__(self, training_history):
        self.actual_revenue = self.initialization.true_dist.get_actual_revenue(self.actual_price)
        self.ideal_price, self.ideal_revenue = self.initialization.true_dist.get_ideals()
        self.regret = self.ideal_revenue - self.actual_revenue
        if self.regret < 0:
            raise ValueError("Regret can never be negative!")


@dataclass
class DOPAuction(Auction):
    pricing_mechanism: str = "DOP"
    actual_price: float = field(init = False)

    def __post_init__(self, training_history):
        self.actual_price = DOP(self.initialization.bids)
        super().__post_init__(training_history)
    

@dataclass
class RSOPAuction(Auction):
    random_seed: int
    pricing_mechanism: str = "RSOP"
    actual_price: float = field(init = False)

    def __post_init__(self, training_history):
        self.actual_price = RSOP(self.initialization.bids, random_seed = self.random_seed)
        super().__post_init__(training_history)


@dataclass
class RSKDEAuction(Auction):
    random_seed: int
    pricing_mechanism: str = "RSKDE"
    actual_price: float = field(init = False)
    estimated_cdfs: tuple[Callable, Callable] = field(init = False)

    def __post_init__(self, training_history):
        self.actual_price, self.estimated_cdfs = RSKDE(self.initialization.bids, 
                                                       lower = self.initialization.true_dist.lower, 
                                                       upper = self.initialization.true_dist.upper,
                                                       random_seed = self.random_seed)
        super().__post_init__(training_history)


@dataclass
class RSRDEAuction(Auction):
    random_seed: int
    training_history: InitVar[robjects.vectors.ListVector]
    RSRDE_method: str = "MLE" # "MLE", "MAP", "BLUP"
    pricing_mechanism: str = "RSRDE"
    actual_price: float = field(init = False)
    estimated_cdfs: tuple[Callable, Callable] = field(init = False)

    def __post_init__(self, training_history):
        self.actual_price, self.estimated_cdfs = RSRDE(self.initialization.bids, 
                                                  lower = self.initialization.true_dist.lower, 
                                                  upper = self.initialization.true_dist.upper, 
                                                  random_seed = self.random_seed,
                                                  method = self.RSRDE_method,
                                                  training_results = training_history)
        super().__post_init__(training_history)


    




            

