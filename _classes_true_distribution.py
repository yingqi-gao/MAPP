from dataclasses import dataclass, field
from typing import Callable, Optional
from functools import partial
from _pricing_utils import max_epc_rev, get_epc_rev
import scipy
import random



@dataclass
class TrueDistribution:
     dist_type: str
     lower: float = 1
     upper: float = 10
     params: dict[str, float] = field(init = False)

     def get_scipy_func(self) -> Callable:
          func_refs = {"uniform": scipy.stats.uniform,
                       "normal": scipy.stats.truncnorm,
                       "exponential": scipy.stats.truncexpon,
                       "pareto": scipy.stats.truncpareto}
          return func_refs[self.dist_type]

     def get_ideals(self) -> tuple[float, float]:
          scipy_func = self.get_scipy_func()
          cdf = partial(scipy_func.cdf, **self.params)
          ideal_price, ideal_revenue = max_epc_rev(cdf, lower = self.lower, upper = self.upper)
          return ideal_price, ideal_revenue
     
     def get_actual_revenue(self, actual_price: float) -> float:
          scipy_func = self.get_scipy_func()
          cdf = partial(scipy_func.cdf, **self.params)
          revenue = get_epc_rev(actual_price, value_cdf = cdf)
          if revenue < 0:
               raise ValueError("Revenue can never be negative!")
          return revenue
     
     def generate_bids(self, num_bidders: int) -> dict[str, float]:
          bids = {}
          for i in range(num_bidders):
               scipy_func = self.get_scipy_func()
               bid = scipy_func.rvs(**self.params)
               if bid < self.lower or bid > self.upper:
                    raise ValueError("Bid generated outside the common support!")
               bids[f"bidder{i + 1}"] = bid
          return bids
          
     
@dataclass
class UniformDistribution(TrueDistribution):
     dist_type: str = "uniform"

     def __post_init__(self):
          self.params = {"loc": self.lower, 
                         "scale": self.upper - self.lower}


@dataclass
class NormalDistribution(TrueDistribution):
     dist_type: str = "normal"
     mean: Optional[float] = None
     sd: Optional[float] = None
     
     def __post_init__(self):
          if self.mean is None:
               self.mean = random.uniform(0, 2 * self.upper)
          if self.sd is None:
               self.sd = random.uniform(0, self.upper)
          self.params = {"a": (self.lower - self.mean) / self.sd,
                         "b": (self.upper - self.mean) / self.sd,
                         "loc": self.mean,
                         "scale": self.sd}


@dataclass
class ExponentialDistribution(TrueDistribution):
     dist_type: str = "exponential"
     scale: Optional[float] = None
     
     def __post_init__(self):
          if self.scale is None:
               self.scale = random.uniform(0, self.upper)
          self.params = {"b": (self.upper - self.lower) / self.scale,
                         "loc": self.lower,
                         "scale": self.scale}


@dataclass
class ParetoDistribution(TrueDistribution):
     dist_type: str = "pareto"
     b: Optional[float] = None
     scale: Optional[float] = None
     
     def __post_init__(self):
          if self.b is None:
               self.b = random.uniform(2, self.upper)
          if self.scale is None:
               self.scale = random.uniform(0, self.upper)
          self.loc = self.lower - self.scale
          self.params = {"b": self.b,
                         "c": (self.upper - self.loc) / self.scale,
                         "loc": self.loc,
                         "scale": self.scale}