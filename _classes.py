from dataclasses import dataclass, field
from typing import Callable, Optional
import scipy
import random
from _pricing_utils import max_epc_rev, get_epc_rev, dict_part
from _py_density_estimation import get_bw
     


@dataclass
class TrueDistribution:
     lower: float
     upper: float
     scipy_func: Callable
     params: dict[str, float] = field(init = False)

     def generate_bids(self, min_size = 10, max_size = 100) -> dict[str, float]:
          size = random.randint(min_size, max_size)
          return {f"bidder{i}": self.scipy_func.rvs(**self.params) for i in range(size)}
     
     def get_ideals(self) -> tuple[float, float]:
          ideal_price = max_epc_rev(self.scipy_func.cdf, lower = self.lower, upper = self.upper, **self.params)
          ideal_revenue = get_epc_rev(ideal_price, value_cdf = self.scipy_func.cdf, **self.params)
          return ideal_price, ideal_revenue
          
     

@dataclass
class UniformDistribution(TrueDistribution):
     scipy_func: Callable = scipy.stats.uniform

     def __post_init__(self):
          self.params = {"loc": self.lower, 
                         "scale": self.upper - self.lower}


@dataclass
class NormalDistribution(TrueDistribution):
     scipy_func: Callable = scipy.stats.truncnorm
     mean: Optional[float] = None
     sd: Optional[float] = None
     
     def __post_init__(self):
          if self.mean is None:
               self.mean = random.uniform(1e-10, 2 * self.upper)
          if self.sd is None:
               self.sd = random.uniform(1e-10, self.upper)
          self.params = {"a": (self.lower - self.mean) / self.sd,
                         "b": (self.upper - self.mean) / self.sd,
                         "loc": self.mean,
                         "scale": self.sd}


@dataclass
class ExponentialDistribution(TrueDistribution):
     scipy_func: Callable = scipy.stats.truncexpon
     scale: Optional[float] = None
     
     def __post_init__(self):
          if self.scale is None:
               self.scale = random.uniform(1e-10, self.upper)
          self.params = {"b": (self.upper - self.lower) / self.scale,
                         "loc": self.lower,
                         "scale": self.scale}



@dataclass
class SingleAuctionBids:
     min_size: int
     max_size: int
     true_dist: TrueDistribution
     bids: dict[str, float] = field(init = False)
     bandwidth: float = field(init = False)
     group1: dict[str, float] = field(init = False)
     group2: dict[str, float] = field(init = False)

     def __post_init__(self):
          self.bids = self.true_dist.generate_bids(self.min_size, self.max_size)
          self.bandwidth = get_bw([self.bids.values()])
          self.group1, self.group2 = dict_part(self.bids)