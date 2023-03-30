import numpy as np
import inspect
import scipy.stats
from typing import *

# Stratified tensor
class StratifiedTensor:
    def __init__(
        self, 
        shape: Tuple[int], 
        dtype: Union[str, np.dtype]=np.float32
    ):
        """
        (d, P, M, *)
        """
        self.X = np.empty(shape, dtype=dtype)
    
    def __getattribute__(self, attr: str):
        try: self.X.__getattribute__(self, attr)
        except: super().__getattribute__(self, attr)
    
    def add(self, index: int, values: np.ndarray):
        """
        assign along group axis
        """
        self.X[:,index] = values

# Histogram
class StratifiedHistogram(StratifiedTensor):
    def __init__(
        self, 
        lo: np.ndarray, 
        hi: np.ndarray,
        num_variables: int, 
        num_groups: int, 
        num_bins: int, 
        dtype: Union[str, np.dtype]=np.float32,
        density: bool=False
    ):
        """Multivariate multigroup histogram: count the number of values for each variable and each group
        Args:
        - lo (np.ndarray[num_variables]): minimum value for each variable
        - hi (np.ndarray[num_variables): maximum value for each variable
        - num_variables (int): number of variables
        - num_groups (int): number of distinct groups
        - num_bins (int): number of histogram bins
        - (optional) dtype (np.dtype): datatype (default: np.float32)
        - (optional) density (bool): Wether to treat as group-wise density (default: False)

        Attributes:
        - histogram (np.ndarray[num_variables, num_groups, num_bins]): group histogram, i.e.
            - H[i,j,k] (float): number of samples in group i and bin j for variable i
        - bins (np.ndarray[num_variables, num_bins])
        """
        self.lo = lo
        self.hi = hi
        self.num_variables, self.num_groups, self.num_bins = num_variables, num_groups, num_bins
        self.histogram = np.zeros((num_variables, num_groups, num_bins), dtype=dtype)
        self.bins = np.stack([np.linspace(self.lo[i], self.hi[i], num_bins) for i in range(num_variables)], axis=0)
    
    def add(self, index: int, values: np.ndarray, others: List[np.ndarray]):
        """Add new group to histogram
        
        Args:
        - values (np.ndarray[num_variables, num_samples]): array of values to

        returns:
        - binned_variance (List[np.ndarray[num_variables, num_groups, num_bins]]): list of num_bins arrays containing variance estimates
        - binned_others (List[List[np.ndarray[num_variables, num_groups, num_bins]]]): list of list of num_bins arrays containing other estimates
        """
        pass

# Metrics
class StratifiedMetric(StratifiedTensor):
    def __init__(self):
        super().__init__(self)
    def evaluate(self, *args, **kwargs):
        """actually compute the metric"""
        raise NotImplementedError()
    def agg(self): raise NotImplementedError()
    def cumagg(self): 
        # loop sum agg
        raise NotImplementedError()
    def ause(self, histogram): 
        # convolution cumagg
        raise NotImplementedError()
    def get_one(self, index):
        results = {str(index): None, str(index)+"_ause": None, str(index)+"_mean": None, str(index)+"_std": None}
        return results
    def get(self):
        # {$pid: values, $pid_ause: values, $pid_mean: values, $pid_std: values, group: nan/values, group_ause: nan/values, group_means: nan/value, group_stds: nan/value} with nan where no sense if len([project_ids])==0
        raise NotImplementedError()
    
class StratifiedMSE(StratifiedMetric): 
    def evaluate(self, *args, **kwargs):
        return super().evaluate(*args, **kwargs)

class StratifiedRMSE(StratifiedMSE):
    def evaluate(self, *args, **kwargs):
        return np.sqrt(super().evaluate(*args, **kwargs))
    
class StratifiedMAE(StratifiedMetric): pass
class StratifiedMBE(StratifiedMetric): pass
class StratifiedUCE(StratifiedMetric): pass
class StratifiedENCE(StratifiedMetric): pass
class StratifiedCIAccuracy(StratifiedMetric): pass
class StratifiedAUCE(StratifiedMetric): pass    

class StratifiedRCU:
    def __init__(
        self,

    ):
        # stratified tensors
        self.histogram = StratifiedHistogram()
        self.mean_variance = StratifiedTensor()
        # stratified metrics
        self.mse = StratifiedMSE()
        self.rmse = StratifiedRMSE()
        self.mae = StratifiedMAE()
        self.mbe = StratifiedMBE()
        self.uce = StratifiedUCE()
        self.ence = StratifiedENCE()
        self.ci_accs = StratifiedCIAccuracy()
        self.auce = StratifiedAUCE()
        # other attributes
        self.index_map = dict()
        self.counter = 0

    @property
    def metrics_tensors(self):
        return dict(
            mse=self.mse,
            rmse=self.rmse,
            mae=self.mae,
            mbe=self.mbe,
            uce=self.uce,
            ence=self.ence,
            ci_accs=self.ci_accs,
            auce=self.auce,
        )    
    
    @property
    def kwargs(self):
        return dict(
            histogram=self.histogram,
            mean_variance=self.mean_variance,
            mean_mse=self.mean_mse,
            empirical_accs=self.empirical_accs,
            index_map=self.index_map,
            counter=self.counter,
        )

    def add_project(self, project_id: str, gt:np.ndarray, mean:np.ndarray, variance:np.ndarray):
        # nan mask flattening
        mask = ~np.isnan(mean).all(0)
        diff = mean[:,mask]-gt[:,mask]
        variance = variance[:,mask]
        # index
        self.index_map[project_id] = self.counter
        self.counter += 1
        index = self.counter
        # add histogram
        p_variance, binned_variance, [binned_diff] = self.histogram.add(index, variance, others=[diff])
        # add mean variance and mean mse
        self.variance.add(index, p_variance)
        # add metrics
        for metric in self.metrics_tensors.values():
            kwargs = {
                k:v for k,v in list(self.kwargs.items())+[("binned_variance", binned_variance), ("binned_diff", binned_diff)] 
                if k in filter(lambda x: x!="self", inspect.getfullargspec(metric.evaluate).args)
            }
            metric.evaluate(**kwargs)

    def get(self, project_ids: Optional[List[str]]=None, metric_names: Optional[List[str]]=None):
        """
        eg rcu.get(EAST)
        """
        assert project_ids is None or isinstance(project_ids, list)
        assert metric_names is None or isinstance(metric_names, list)
        # get indexes and metrics
        indexes = self.index_map.values() if project_ids is None else [self.index_map[pid] for pid in project_ids]
        metrics = self.metrics_tensors.values() if metric_names is None else [self.metrics_tensor[mid] for mid in metric_names]
        # compute
        results = []
        for metric in metrics:
            results.append(metric.get(project_ids)) # {$pid: values, group: values, ause: values, group_means: value, group_stds: value}
        return results
    
def main():
    """
    - load standardization data
    - loop on projects, compute variance bounds online
    - init rcu
    - loop on projects
        - standardize
        - add project
        - get([project_id])
    - loop on regions
        - get(region)
    - save results (incl. histogram)
    """
    pass
