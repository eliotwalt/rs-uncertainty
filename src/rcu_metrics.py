import numpy as np
import inspect
import scipy.stats
from typing import *

# Utils
def weighted_avg(values, counts, axis):
    return np.nansum(values*counts, axis=axis)/np.nansum(counts, axis=axis)

# stratified tensor parent
class StratifiedTensor:
    def __init__(
        self, 
        num_variables: int, 
        num_groups: int, 
        num_bins: int,
        dtype: Union[str, np.dtype]=np.float32
    ):
        """
        (d, P, M, *)
        """
        self.num_variables, self.num_groups, self.num_bins = num_variables, num_groups, num_bins
        self.X = np.nan*np.ones((num_variables, num_groups, num_bins), dtype=dtype)
        self.variables_axis, self.groups_axis, self.bins_axis = 0, 1, 2

    @property
    def array(self): return self.X
    
    def __getattribute__(self, attr: str):
        try: self.X.__getattribute__(self, attr)
        except: super().__getattribute__(self, attr)
    
    def add(self, index: int, values: np.ndarray):
        """
        assign along group axis
        """
        self.X[:,index] = values

# histogram
class StratifiedHistogram(StratifiedTensor):
    def __init__(
        self, 
        lo: np.ndarray, 
        hi: np.ndarray, 
        *args, **kwargs
    ):
        """Multivariate multigroup histogram: count the number of values for each variable and each group
        Args:
        - lo (np.ndarray[num_variables]): minimum value for each variable
        - hi (np.ndarray[num_variables): maximum value for each variable

        Attributes:
        - X(np.ndarray[num_variables, num_groups, num_bins]): group histogram, i.e.
            - H[i,j,k] (float): number of samples in group i and bin j for variable i
        - bins (np.ndarray[num_variables, num_bins])
        """
        super().__init__(self, *args, **kwargs)
        self.lo = lo
        self.hi = hi
        self.bins = np.stack([np.linspace(self.lo[i], self.hi[i], self.num_bins) for i in range(self.num_variables)], axis=self.variables_axis)
    
    def add(self, index: int, values: np.ndarray, others: List[np.ndarray]):
        """Add new group to histogram
        
        Args:
        - values (np.ndarray[num_variables, num_samples]): array of values to

        returns:
        - binned_variance (List[np.ndarray[num_variables, num_samples]]): list of num_bins arrays containing variance estimates
        - binned_others (List[List[np.ndarray[num_variables, num_samples]]]): list of list of num_bins arrays containing other estimates
        """
        pass

# metrics Parent
class StratifiedMetric(StratifiedTensor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def evaluate(self, *args, **kwargs):
        """actually compute the metric"""
        raise NotImplementedError()
    
    def agg(self, *args, **kwargs): 
        """aggregate"""
        raise NotImplementedError()
    
    def cumagg(self, histogram, arr=None, cumaxis=None, aggaxis=None):
        """cumulatively aggregate"""
        if arr is None: arr = self.X
        if cumaxis is None: cumaxis = self.bins_axis
        if aggaxis is None: aggaxis = self.groups_axis
        assert cumaxis != aggaxis
        cumX = np.nan*np.ones((self.num_variables, self.num_bins))
        cumH = np.nancumsum(histogram, axis=cumaxis)
        for k in range(self.num_bins):
            cumX[:,k] = self.agg(cumH[:,:,:k], arr[:,:,:k], axis=aggaxis)
        return cumX
    
    def ause(self, histogram, arr=None): 
        cumX = self.cumagg(histogram, arr, cumaxis=self.bins_axis, aggaxis=self.groups_axis)
        return np.nansum((cumX[:,1:]+cumX[:,:-1]), axis=self.bins_axis-1)/(2*self.num_bins)

    def get(self, histogram, arr=None):
        results = {"values": None, "ause": None, "std": None}
        if arr is None: arr = self.X
        results["values"] = self.agg(histogram, axis=(self.groups_axis, self.bins_axis))
        results["ause"] = self.ause(histogram)
        results["std"] = np.nanstd(self.X, axis=(self.groups_axis, self.bins_axis))
        return results
    
    def get_subset(self, histogram, indexes):
        subX = self.X[indexes,:]
        return self.get(histogram, subX)

# regression metrics parent
class StratifiedMeanErrorMetric(StratifiedMetric):
    def __init__(self, fn: Callable, *args, **kwargs):
        self.fn = fn
        super().__init__(*args, **kwargs)
    def evaluate(self, binned_diff):
        return np.stack([np.nanmean(self.fn(bd), axis=1) if bd.shape[1]>0 else np.nan*np.ones(bd.shape[0],) for bd in binned_diff], axis=1)
    def agg(self, histogram, arr=None, axis=None):
        if arr is None: arr = self.X
        if axis is None: axis = self.groups_axis
        return weighted_avg(arr, histogram, axis=axis)
    
# regression metrics
class StratifiedMSE(StratifiedMetric): 
    def __init__(self, *args, **kwargs): 
        super().__init__(self, fn=lambda x: x**2)
class StratifiedRMSE(StratifiedMSE):
    def evaluate(self, binned_diff):
        return np.sqrt(super().evaluate(binned_diff))    
class StratifiedMAE(StratifiedMetric):
    def __init__(self, *args, **kwargs): 
        super().__init__(self, fn=lambda x: np.abs(x))
class StratifiedMBE(StratifiedMetric):
    def __init__(self, *args, **kwargs): 
        super().__init__(self, fn=lambda x: x)

# calibration metrics
class StratifiedUCE(StratifiedMetric): pass
class StratifiedENCE(StratifiedMetric): pass
class StratifiedCIAccuracy(StratifiedMetric): pass
class StratifiedAUCE(StratifiedMetric): pass

# Usefulness metrics
class StratifiedCv(StratifiedMetric): pass
class StratifiedSRP(StratifiedMetric): pass 

class StratifiedRCU:
    def __init__(
        self,
        # shapes
        num_variables: int, 
        num_groups: int, 
        num_bins: int,
        # histogram
        lo_variance: np.ndarray,
        hi_variance: np.ndarray,
        # ci_accs
        lo_rho: float,
        hi_rho: float,
        num_rhos: int=100,
    ):
        shape_kw={num_variables: num_variables, num_groups: num_groups, num_variables: num_variables}
        # stratified tensors
        self.histogram = StratifiedHistogram(lo_variance, hi_variance, **shape_kw)
        self.mean_variance = StratifiedTensor(**shape_kw)
        # stratified metrics
        self.mse = StratifiedMSE(**shape_kw)
        self.rmse = StratifiedRMSE(**shape_kw)
        self.mae = StratifiedMAE(**shape_kw)
        self.mbe = StratifiedMBE(**shape_kw)
        self.uce = StratifiedUCE(**shape_kw)
        self.ence = StratifiedENCE(**shape_kw)
        self.ci_accs = StratifiedCIAccuracy(num_rhos=num_rhos, lo=lo_rho, hi=hi_rho, **shape_kw)
        self.ci90_accs = StratifiedCIAccuracy(num_rhos=1, lo=0.9, hi=0.9, **shape_kw)
        self.auce = StratifiedAUCE(**shape_kw)
        self.cv = StratifiedCv(**shape_kw)
        self.srp = StratifiedSRP(**shape_kw)
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
            ci90_accs=self.ci90_accs,
            auce=self.auce,
        )    
    
    @property
    def kwargs(self):
        return dict(
            histogram=self.histogram,
            mean_variance=self.mean_variance,
            mean_mse=self.mean_mse,
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
            metric.add(index, metric.evaluate(**kwargs))
    
    def get(self, metric_names: Optional[List[str]]=None):
        """eg rcu.get(["mse", "mbe"])"""
        # get metrics
        if metric_names is None: metric_names = list(self.metrics_tensors.keys())
        metrics = [self.metrics_tensor[mid] for mid in metric_names]
        # compute
        results = {}
        for metric, metric_name in zip(metrics, metric_names):
            results[metric_name] = metric.get(self.histogram)
        return results

    def get_subset(self, project_ids: Optional[List[str]]=None, metric_names: Optional[List[str]]=None):
        """eg rcu.get_multi(EAST, ["mse", "uce"])"""
        assert project_ids is None or isinstance(project_ids, list)
        assert metric_names is None or isinstance(metric_names, list)
        # get indexes and metrics
        indexes = self.index_map.values() if project_ids is None else [self.index_map[pid] for pid in project_ids]
        metrics = self.metrics_tensors.values() if metric_names is None else [self.metrics_tensor[mid] for mid in metric_names]
        # compute
        results = []
        for metric in metrics:
            results.append(metric.get_subset(self.histogram, indexes))
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
