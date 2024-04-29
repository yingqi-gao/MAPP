from dataclasses import dataclass, field
from typing import Callable, Optional
from functools import partial
from _pricing_utils import max_epc_rev, get_epc_rev
import scipy
import random
from _truncated_lognormal import log_normal_truncated_ab_cdf, log_normal_truncated_ab_sample



@dataclass
class TrueDistribution:
     lower: float
     upper: float
     dist_type: str
     scipy_func: Callable
     sample_func: Optional[Callable] = None
     params: dict[str, float] = field(init = False)
     cdf: Callable = field(init = False)
     ideal_price: float = field(init = False)
     ideal_revenue: float = field(init = False)

     def __post_init__(self):
          if self.dist_type == "lognormal":
               self.cdf = partial(self.scipy_func, **self.params)
          else:
               self.cdf = partial(self.scipy_func.cdf, **self.params)
          self.ideal_price, self.ideal_revenue = max_epc_rev(self.cdf, lower = self.lower, upper = self.upper)

     def generate_bids(self, num_bidders: int) -> dict[str, float]:
          bids = {}
          for i in range(num_bidders):
               if self.dist_type == "lognormal":
                    bid = self.sample_func(**self.params)
               else:
                    bid = self.scipy_func.rvs(**self.params)
               if bid < self.lower or bid > self.upper:
                    raise ValueError("Bid generated outside the common support!")
               bids[f"bidder{i}"] = bid
          return bids
     
     def get_actual_revenue(self, actual_price: float) -> float:
          revenue = get_epc_rev(actual_price, value_cdf = self.cdf)
          if revenue < 0:
               raise ValueError("Revenue can never be negative!")
          return revenue
          
     
     
@dataclass
class UniformDistribution(TrueDistribution):
     dist_type: str = "uniform"
     scipy_func: Callable = scipy.stats.uniform

     def __post_init__(self):
          self.params = {"loc": self.lower, 
                         "scale": self.upper - self.lower}
          super().__post_init__()


@dataclass
class NormalDistribution(TrueDistribution):
     dist_type: str = "normal"
     scipy_func: Callable = scipy.stats.truncnorm
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
          super().__post_init__()


@dataclass
class ExponentialDistribution(TrueDistribution):
     dist_type: str = "exponential"
     scipy_func: Callable = scipy.stats.truncexpon
     scale: Optional[float] = None
     
     def __post_init__(self):
          if self.scale is None:
               self.scale = random.uniform(0, self.upper)
          self.params = {"b": (self.upper - self.lower) / self.scale,
                         "loc": self.lower,
                         "scale": self.scale}
          super().__post_init__()


@dataclass
class ParetoDistribution(TrueDistribution):
     dist_type: str = "pareto"
     scipy_func: Callable = scipy.stats.truncpareto
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
          super().__post_init__()


@dataclass
class LognormalDistribution(TrueDistribution):
     dist_type: str = "lognormal"
     scipy_func: Callable = log_normal_truncated_ab_cdf
     sample_func: Callable = log_normal_truncated_ab_sample
     mu: Optional[float] = None
     sigma: Optional[float] = None
     
     def __post_init__(self):
          if self.mu is None:
               self.mu = random.uniform(0, 2 * self.upper)
          if self.sigma is None:
               self.sigma = random.uniform(0, self.upper)
          self.params = {"mu": self.mu,
                         "sigma": self.sigma,
                         "a": self.lower,
                         "b": self.upper}
          super().__post_init__()