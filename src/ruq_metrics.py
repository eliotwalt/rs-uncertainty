import numpy as np
from itertools import chain
import scipy.stats
import yaml
from copy import deepcopy

# Utils
def weighted_avg(values, counts, axis):
    return np.nansum(values*counts, axis=axis)/np.nansum(counts, axis=axis)

class MutlivariateGroupHistogram:
    def __init__(
        self, 
        lo, 
        hi,
        num_variables, 
        num_groups, 
        num_bins, 
        dtype=np.float32
    ):
        """Multivariate group histogram: count the number of values for each variable and each group
        Args:
        - lo (np.ndarray[num_variables]): minimum value for each variable
        - hi (np.ndarray[num_variables): maximum value for each variable
        - num_variables (int): number of variables
        - num_groups (int): number of distinct groups
        - num_bins (int): number of histogram bins
        - (optional) dtype (np.dtype): datatype (default: np.float32)

        Attributes:
        - histogram (np.ndarray[num_variables, num_groups, num_bins]): group histogram, i.e.
            - H[i,j,k] (float): number of samples in group i and bin j for variable i
        - bins (np.ndarray[num_variables, num_bins])
        """
        self.lo = lo
        self.hi = hi
        self.num_variables, self.num_groups, self.num_bins = num_variables, num_groups, num_bins
        self.histogram = np.zeros((num_variables, num_groups, num_bins), dtype=dtype)
        self.group_cursor = 0
        self.bins = np.stack([np.linspace(self.lo[i], self.hi[i], num_bins) for i in range(num_variables)], axis=0)
    
    def add(self, values):
        """Add new group to histogram
        
        Args:
        - values (np.ndarray[num_variables, num_samples]): array of values to 
        """
        assert self.group_cursor < self.num_groups, "Histogram is full"
        self.group_cursor += 1
        for vid in range(self.num_variables):
            vvalues = values[vid]
            bin_ids = np.digitize(vvalues, bins=self.bins[vid])-1

    
    def __getitem__(self, selection):
        return self.H[selection]

# Base metric: no need to implement empty init each time
class BaseMetric:
    def __init__(self): pass
    def __call__(self, *args, **kwargs): raise NotImplementedError()
    def agg(self, *args, **kwargs): raise NotImplementedError()
    def cumagg(self, *args, **kwargs): raise NotImplementedError()

# Regression metrics: MSE, RMSE, MAE, MBE
class BaseMeanError(BaseMetric):
    def __init__(self, function): self.function = function
    def __call__(self, diff):
        """Compute metric on single set of samples
        
        Args:
        - diff (np.ndarray[num_variables, num_samples]): sample-wise difference (ground_truth - predicted_mean)
        
        Returns:
        - values (np.ndarray[num_variables,]): metric value for each variable
        """
        assert len(diff.shape) == 2
        num_variables, _ = diff.shape
        values = np.nanmean(self.function(diff), axis=1)
        assert values.shape == (num_variables, )
        return values
    def agg(self, values, counts):
        """Aggregate metrics across multiple groups
        
        Args:
        - values (np.ndarray[num_variables, num_groups]): metric values for each group
        - counts (np.ndarray[num_groups]): number of samples in each group

        Returns:
        - counts 
        """
        assert len(values.shape) == 2
        assert len(counts.shape) == 1
        assert values.shape[1]==counts.shape[0]
        num_variables, _ = values.shape
        counts = np.expand_dims(counts, axis=0) # (1, num_groups)
        values = weighted_avg(values, counts, axis=1)
        assert values.shape == (num_variables, )
        return values
    def cumagg(self, values, counts):
        """Cumulatively aggregate metrics across multiple groups
        
        Args:
        - values (np.ndarray[num_variables, num_groups, num_bins]): metric values for each groups and each bin
        - counts (np.ndarray[num_groups, num_bins]): number of samples in each group and each bin

        Returns:
        - counts 
        """


class MSE(BaseMeanError):
    def __init__(self): super().__init__(self, function=lambda x: x**2)

class RMSE(MSE):
    def __call__(self, diff): return np.sqrt(super().__call__(diff))
    def agg(self, values, counts): return np.sqrt(super().agg(values, counts))

class MAE(BaseMeanError):
    def __init__(self): super().__init__(self, function=np.abs)

class MBE(BaseMeanError):
    def __init__(self): super().__init__(self, function=lambda x: x)    

# Calibration Metrics: UCE, ENCE, AUCE, AUSE
class UCE(BaseMetric):
    def __init__(self, )
        
# Usefullness metrics: Cv, R90

# MetricsTensor
class MetricsTensor:
    def __init__(self, shape, metric):
        self.tensor = np.zeros(shape)
        self.metric = metric
    
    def add(self, index, )

# RunningRCU
class RunningRCU:
    """
    """
    def __init__(
        self, 
        # histogram
        lo, 
        hi,
        num_variables, # d
        num_groups, # P
        num_bins, # M
    ):
        # attributes
        self.num_variables, self.num_groups, self.num_bins = num_variables, num_groups, num_bins
        # metrics
        self.init_metrics_fns()
        # fine grained arrays
        self.mses = self.zeros()
        self.rmses = self.zeros()
        self.maes = self.zeros()
        self.mbes = self.zeros()
        self.uces = self.uces()
        self.
        self.histogram = self.MultivariateGroupHistogram(lo, hi, num_variables, num_groups, num_bins) # h
        self.mean_variances = self.zeros() # \bar\sigma^2
        self.
    
    def init_metric_fns(self):
        self.mse_fn = MSE()
        self.rmse_fn = RMSE()
        self.mae_fn = MAE()
        self.mbe_fn = MBE()
        self.uce_fn = UCE()
        self.ence_fn = ENCE()
        self.auce_fn = AUCE()
        self.rho_fn = ConfidenceAccuracy()
        self.ause_fn = AUSE()
        self.cv_fn = Cv()
        self.srp_fn = SRP()
    
    def init_metrics_tensors(self):
        zeros = lambda: np.zeros((self.num_variables, self.num_groups, self.num_bins))
        self.mse = zeros()
        self.rmse = zeros()
        self.mae = zeros()
        self.mbe = zeros()
        self.uce = zeros()
        self.ence = zeros()
        self.auce = zeros()
        self.rho = zeros()
        self.ause = zeros()
        self.cv = zeros()
        self.srp = zeros()
