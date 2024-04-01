from dataclasses import dataclass, field, InitVar
from _classes_auction_initialization import AuctionInitialization, OnlineAuctionRandomInitialization
from typing import Optional, Tuple, Callable
from _pricing_mechanisms import DOP, RSOP, RSKDE, RSRDE
from _pricing_utils import scale_value, scale_cdf
from _py_density_estimation import get_bw



@dataclass
class AuctionResults:
    initialization: InitVar[AuctionInitialization]
    actual_price: float
    pricing_mechanism: str
    actual_revenue: float = field(init = False)
    ideal_price: float = field(init = False)
    ideal_revenue: float = field(init = False)
    regret: float = field(init = False)
    estimated_cdfs: Optional[Tuple[Callable, Callable]] = None
    RSRDE_method: Optional[str] = None

    def __post_init__(self, initialization):
        self.actual_revenue = initialization.true_dist.get_actual_revenue(self.actual_price)
        self.ideal_price, self.ideal_revenue = initialization.true_dist.get_ideals()
        self.regret = self.ideal_revenue - self.actual_revenue


@dataclass
class DOPResults(AuctionResults):
    pricing_mechanism: str = "DOP"


@dataclass
class RSOPResults(AuctionResults):
    pricing_mechanism: str = "RSOP"


@dataclass
class RSKDEResults(AuctionResults):
    estimated_cdfs: Tuple[Callable, Callable]
    pricing_mechanism: str = "RSKDE"
    

@dataclass
class RSRDEResults(AuctionResults):
    RSRDE_method: str # "MLE", "MAP", "BLUP"
    estimated_cdfs: Tuple[Callable, Callable]
    pricing_mechanism: str = "RSRDE"
    


@dataclass
class OnlineAuctions:
    online_initialization: OnlineAuctionRandomInitialization
    DOP_results: list[DOPResults] = field(init = False)
    RSOP_results: list[RSOPResults] = field(init = False)
    RSKDE_results: list[RSKDEResults] = field(init = False)
    RSRDE_results: list[RSRDEResults] = field(init = False) # half the length of other's

    def __post_init__(self):
        is_upper_floated = self.online_initialization.is_upper_floated
        common_upper = self.online_initialization.upper

        DOP_results, RSOP_results, RSKDE_results = [], [], []
        RSRDE_MLE_results, RSRDE_MAP_results, RSRDE_BLUP_results = [], [], []
        train_hist, train_bws = [], []

        for i in range(self.online_initialization.num_rounds):
            auction_initialization = self.online_initialization.sequence_auctions[i]
            bids = auction_initialization.bids
            lower = auction_initialization.true_dist.lower
            upper = auction_initialization.true_dist.upper

            # DOP
            DOP_price = DOP(bids)
            DOP_results.append(DOPResults(initialization = auction_initialization, actual_price = DOP_price))
            
            # RSOP
            RSOP_price = RSOP(bids)
            RSOP_results.append(RSOPResults(initialization = auction_initialization, actual_price = RSOP_price))
            
            # RSKDE
            RSKDE_price, RSKDE_estimated_cdfs = RSKDE(bids, lower = lower, upper = upper)
            RSKDE_results.append(RSKDEResults(initialization = auction_initialization, actual_price = RSKDE_price, estimated_cdfs = RSKDE_estimated_cdfs))
            
            # RSRDE
            if i >= 100:
                # MLE
                RSRDE_MLE_price, RSRDE_MLE_estimated_cdfs = RSRDE(bids, lower = lower, upper = common_upper, train_hist = train_hist, train_bws = train_bws, method = "MLE")
                RSRDE_MLE_results.append(RSRDEResults(initialization = auction_initialization, 
                                                      actual_price = RSRDE_MLE_price if not is_upper_floated else scale_value(RSRDE_MLE_price, lower = lower, old_upper = common_upper, new_upper = upper), 
                                                      estimated_cdfs = RSRDE_MLE_estimated_cdfs if not is_upper_floated else scale_cdf(RSRDE_MLE_estimated_cdfs, lower = lower, old_upper = common_upper, new_upper = upper), 
                                                      RSRDE_method = "MLE"))
                # MAP
                RSRDE_MAP_price, RSRDE_MAP_estimated_cdfs = RSRDE(bids, lower = lower, upper = common_upper, train_hist = train_hist, train_bws = train_bws, method = "MAP")
                RSRDE_MAP_results.append(RSRDEResults(initialization = auction_initialization, 
                                                      actual_price = RSRDE_MAP_price if not is_upper_floated else scale_value(RSRDE_MAP_price, lower = lower, old_upper = common_upper, new_upper = upper), 
                                                      estimated_cdfs = RSRDE_MAP_estimated_cdfs if not is_upper_floated else scale_cdf(RSRDE_MAP_estimated_cdfs, lower = lower, old_upper = common_upper, new_upper = upper),  
                                                      RSRDE_method = "MAP"))
                # BLUP
                RSRDE_BLUP_price, RSRDE_BLUP_estimated_cdfs = RSRDE(bids, lower = lower, upper = common_upper, train_hist = train_hist, train_bws = train_bws, method = "BLUP")
                RSRDE_BLUP_results.append(RSRDEResults(initialization = auction_initialization, 
                                                       actual_price = RSRDE_BLUP_price if not is_upper_floated else scale_value(RSRDE_BLUP_price, lower = lower, old_upper = common_upper, new_upper = upper), 
                                                       estimated_cdfs = RSRDE_BLUP_estimated_cdfs if not is_upper_floated else scale_cdf(RSRDE_BLUP_estimated_cdfs, lower = lower, old_upper = common_upper, new_upper = upper),  
                                                       RSRDE_method = "BLUP"))
            
            bids_list = [bid if not is_upper_floated else scale_value(bid, lower = lower, old_upper = upper, new_upper = common_upper) for bid in bids.values()]
            train_hist.append(bids_list)
            train_bws.append(get_bw(bids_list))
            


            

