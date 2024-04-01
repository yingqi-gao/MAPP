from dataclasses import dataclass, InitVar, field
from _classes_true_distribution import TrueDistribution, UniformDistribution, NormalDistribution, ExponentialDistribution
from typing import Optional
import random



@dataclass
class AuctionInitialization:
     true_dist: TrueDistribution
     num_bidders: Optional[int] = None
     min_num_bidders: InitVar[Optional[int]] = None
     max_num_bidders: InitVar[Optional[int]] = None
     bids: dict[str, float] = field(init = False)

     def __post_init__(self, min_num_bidders, max_num_bidders):
          if all(item is None for item in [self.num_bidders, min_num_bidders, max_num_bidders]):
               raise ValueError("You must specify the number of bidders or its range.")
          self.num_bidders = random.randint(min_num_bidders, max_num_bidders) if self.num_bidders is None else self.num_bidders
          self.bids = self.true_dist.generate_bids(self.num_bidders)




@dataclass
class OnlineAuctionRandomInitialization:
     num_rounds: int
     distribution_type: str
     lower: float
     upper: float
     is_upper_floated: bool
     min_num_bidders: int
     max_num_bidders: int
     sequence_auctions: list[AuctionInitialization] = field(init = False)

     def __post_init__(self):
          sequence_auctions = []
          distribution_reference = {
               "uniform": UniformDistribution,
               "normal": NormalDistribution,
               "exponential": ExponentialDistribution
          }
          for i in range(self.num_rounds):
               upper = random.uniform(0.5 * self.upper, 2 * self.upper) if self.is_upper_floated else self.upper
               true_dist = distribution_reference[self.distribution_type](lower = self.lower, upper = upper)
               auction_initialization_at_t = AuctionInitialization(true_dist = true_dist, min_num_bidders = self.min_num_bidders, max_num_bidders = self.max_num_bidders)
               sequence_auctions.append(auction_initialization_at_t)
          self.sequence_auctions = sequence_auctions




