from dataclasses import dataclass, field, InitVar
from _py_density_estimation import get_bw
from _classes_initialization import AuctionInitialization
from typing import Optional, Callable
from _pricing_mechanisms import DOP, RSOP, RSKDE, RSRDE
from _pricing_utils import scale_cdf
import warnings


@dataclass
class TrainingData: 
    observations: list[float]
    bandwidth: float = field(init = False)

    def __post_init__(self):
        self.bandwidth = get_bw(self.observations)



@dataclass
class Auction:
    initialization: AuctionInitialization
    pricing_mechanism: str
    actual_price: float
    actual_revenue: float = field(init = False)
    ideal_price: float = field(init = False)
    ideal_revenue: float = field(init = False)
    regret: float = field(init = False)
    common_upper: InitVar[Optional[float]] = None
    is_upper_floated: InitVar[Optional[bool]] = None
    training_history: InitVar[Optional[list[TrainingData]]] = None

    def __post_init__(self, common_upper, is_upper_floated, training_history):
        self.actual_revenue = self.initialization.true_dist.get_actual_revenue(self.actual_price)
        self.ideal_price, self.ideal_revenue = self.initialization.true_dist.get_ideals()
        self.regret = self.ideal_revenue - self.actual_revenue
        if self.regret < 0:
            print()
            raise ValueError("Regret can never be negative!")

@dataclass
class DOPAuction(Auction):
    pricing_mechanism: str = "DOP"
    actual_price: float = field(init = False)

    def __post_init__(self, common_upper, is_upper_floated, training_history):
        self.actual_price = DOP(self.initialization.bids)
        super().__post_init__(common_upper, is_upper_floated, training_history)
    

@dataclass
class RSOPAuction(Auction):
    pricing_mechanism: str = "RSOP"
    actual_price: float = field(init = False)

    def __post_init__(self, common_upper, is_upper_floated, training_history):
        self.actual_price = RSOP(self.initialization.bids)
        super().__post_init__(common_upper, is_upper_floated, training_history)


@dataclass
class RSKDEAuction(Auction):
    pricing_mechanism: str = "RSKDE"
    actual_price: float = field(init = False)
    estimated_cdfs: tuple[Callable, Callable] = field(init = False)

    def __post_init__(self, common_upper, is_upper_floated, training_history):
        self.actual_price, self.estimated_cdfs = RSKDE(self.initialization.bids, 
                                                       lower = self.initialization.true_dist.lower, 
                                                       upper = self.initialization.true_dist.upper)
        super().__post_init__(common_upper, is_upper_floated, training_history)


@dataclass
class RSRDEAuction(Auction):
    common_upper: InitVar[float]
    is_upper_floated: InitVar[bool]
    training_history: InitVar[list[TrainingData]]
    RSRDE_method: str = "MLE" # "MLE", "MAP", "BLUP"
    pricing_mechanism: str = "RSRDE"
    actual_price: float = field(init = False)
    estimated_cdfs: tuple[Callable, Callable] = field(init = False)

    def __post_init__(self, common_upper, is_upper_floated, training_history):
        self.actual_price, estimated_cdfs = RSRDE(self.initialization.bids, 
                                                  lower = self.initialization.true_dist.lower, 
                                                  upper = common_upper, 
                                                  train_hist = [training_data.observations for training_data in training_history], 
                                                  train_bws = [training_data.bandwidth for training_data in training_history], 
                                                  method = self.RSRDE_method)
        if is_upper_floated:
            lower = self.initialization.true_dist.lower
            upper = self.initialization.true_dist.upper
            if upper == common_upper:
                    warnings.warn(f"Upper is the same as the common upper: {common_upper}")
            self.estimated_cdfs = scale_cdf(estimated_cdfs, lower = lower, old_upper = common_upper, new_upper = upper)
        super().__post_init__(common_upper, is_upper_floated, training_history)


    




            

