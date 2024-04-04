from dataclasses import dataclass, field
from typing import Callable, Optional
from functools import partial
from _pricing_utils import max_epc_rev, get_epc_rev
import scipy
import random



@dataclass
class TrueDistribution:
     lower: float
     upper: float
     scipy_func: Callable
     params: dict[str, float] = field(init = False)
     cdf: Callable = field(init = False)

     def __post_init__(self):
          self.cdf = partial(self.scipy_func.cdf, **self.params)

     def generate_bids(self, num_bidders: int) -> dict[str, float]:
          return {f"bidder{i}": self.scipy_func.rvs(**self.params) for i in range(num_bidders)}
     
     def get_ideals(self) -> tuple[float, float]:
          return max_epc_rev(self.cdf, lower = self.lower, upper = self.upper)
     
     def get_actual_revenue(self, actual_price: float) -> float:
          return get_epc_rev(actual_price, value_cdf = self.cdf)
          
     
     
@dataclass
class UniformDistribution(TrueDistribution):
     scipy_func: Callable = scipy.stats.uniform

     def __post_init__(self):
          self.params = {"loc": self.lower, 
                         "scale": self.upper - self.lower}
          super().__post_init__()


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
          super().__post_init__()


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
          super().__post_init__()