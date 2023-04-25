import numpy as np
import inspect
import json
import scipy.stats
from itertools import chain
from typing import *
from math import ceil
from copy import deepcopy
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
sns.set_style("whitegrid")

# Utils
def arrequal(a, b):
    if np.isnan(a).sum()==a.flatten().shape[0] and np.isnan(b).sum()==b.flatten().shape[0]:
        return True # full nan arrays
    try: np.testing.assert_equal(a,b)
    except: return False
    return True

def nan_frac(arr):
    if arr.shape==(0,): return 0.
    return np.isnan(arr).sum()/(np.isnan(arr).sum()+(~np.isnan(arr)).sum())

def weighted_avg(values, counts, axis, keepdims=False):
    return np.nansum(values*counts, axis=axis, keepdims=keepdims)/np.nansum(counts, axis=axis, keepdims=keepdims)

def reduce_binned(binned, reduce_fn, *args):
    num_variables, num_bins = len(binned), len(binned[0])
    result = np.full((num_variables, num_bins), np.nan)
    for i in range(num_variables):
        for j in range(num_bins):
            if binned[i][j].shape==(0,): continue
            result[i,j] = reduce_fn(binned[i][j], *args) # reduce list binned[i][j] into single value
    return result

# stratified tensor parent
class StratifiedTensor():
    def __init__(
        self, 
        num_tensors: int,
        num_variables: int, 
        num_groups: int, 
        num_bins: int,
        dtype: Union[str, np.dtype]=np.float32
    ):
        """
        (T, d, P, M, *)
        """
        self.X = np.nan*np.ones((num_tensors, num_variables, num_groups, num_bins), dtype=dtype)
        self.num_tensors, self.num_variables, self.num_groups, self.num_bins = self.X.shape
        self.variables_axis, self.groups_axis, self.bins_axis = 0, 1, 2 # axis in each individual tensor, i.e. in X[i]
    def __getitem__(self, sel): return self.X.__getitem__(sel)
    @property
    def array(self): return self.X
    @property
    def dtype(self): return self.X.dtype    
    @property
    def shape(self): return self.X.shape
    def add(self, index: int, *values: np.ndarray):
        """assign along group axis"""
        for i, vals in enumerate(values):
            self.X[i,:,index] = vals
    def __eq__(self, other):
        if type(other)!=type(self): return False
        else: 
            if not arrequal(self.X,other.X): return False 
        return True

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
        super().__init__(1, *args, **kwargs)
        self.lo = lo
        self.hi = hi
        # linear binning
        self.bins = np.stack([
            np.linspace(self.lo[i], self.hi[i], self.num_bins) 
            for i in range(self.num_variables)
        ], axis=self.variables_axis)
    
    def add(self, index: int, values: np.ndarray, others: List[np.ndarray]=None):
        """Add new group to histogram
        
        Args:
        - values (np.ndarray[num_variables, num_samples]): array of values to

        returns:
        - mean_values (np.ndarray[num_variables, num_bins]): mean values for each bin and each variable
        - binned_values (List[np.ndarray[num_variables, num_samples]]): list of num_variables arrays containing values of each bin
        - binned_others (List[List[np.ndarray[num_variables, num_samples]]]): list of list of num_variables arrays containing other values in each bin
        """
        # X: (vars, groups, bin) -> add to X[:,index,:]        
        binned_values = [[[] for _ in range(self.num_bins)] for _ in range(self.num_variables)]
        if others is not None: 
            assert isinstance(others, list), "others must be a list of arrays, got {}".format(type(others))
            binned_others = [[[[] for _ in range(self.num_bins)] for _ in range(self.num_variables)] for _ in range(len(others))]
        # allocate to bins
        bins_ids = np.stack(
            [np.digitize(d_values, bins=self.bins[i])-1 
            for i, d_values in enumerate(values)], axis=0
        )
        # bin loop
        for bin_id in range(self.num_bins):
            # mask
            mask = (bins_ids==bin_id)
            for i, variable_mask in enumerate(mask):
                self.X[0,:,index,bin_id] = mask.sum(axis=1)
                binned_values[i][bin_id] = values[i,variable_mask]
                for j, other in enumerate(others): 
                    binned_others[j][i][bin_id] = others[j][i,variable_mask]
        # compute mean values
        mean_values = reduce_binned(binned_values, reduce_fn=lambda x: np.nanmean(x, axis=0))
        return mean_values, binned_values, binned_others
    
# metrics Parent
class StratifiedMetric(StratifiedTensor):
    def evaluate_binned(self, *args, **kwargs):
        """compute the metric in each bin and each variables"""
        raise NotImplementedError()

    def compute(self, *args, **kwargs):
        """actually compute the metric in a given set of variables"""
        raise NotImplementedError()

    def agg(self, *args, **kwargs): 
        """aggregate (d,P,M)->(d,) if not keepbins else (d,P,M)->(d,M)"""
        raise NotImplementedError()

    def agg_tensor(self, H, bins):
        X = self.X[...,bins]
        return weighted_avg(X, H, axis=-1, keepdims=True)
    
    def cumagg(self, histogram, *arrs):
        cummetric = np.nan*np.ones((self.num_variables, self.num_bins))
        cumH = np.nancumsum(histogram, axis=self.groups_axis)
        for k in range(1,self.num_bins):
            binarrs = (arr[:,:,:k] for arr in arrs)
            cummetric[:,k] = self.agg(cumH[:,:,:k], *binarrs).reshape(cummetric[:,k].shape)
        return cummetric

    def ause(self, histogram, *arrs): 
        cummetric = self.cumagg(histogram, *arrs)
        return np.nansum((cummetric[:,1:]+cummetric[:,:-1]), axis=self.bins_axis-1)/(2*self.num_bins)

    def get(self, histogram, *arrs):
        if not isinstance(histogram, np.ndarray): histogram = histogram.array[0]
        results = {"values": None, "ause": None}
        if arrs==(): arrs = tuple(self.X[i] for i in range(self.num_tensors))
        results["values"] = self.agg(histogram, *arrs)
        results["ause"] = self.ause(histogram, *arrs)
        return results
    
    def get_subset(self, histogram, indexes):
        if not isinstance(histogram, np.ndarray): histogram = histogram.array[0]
        histogram = histogram[:,indexes]
        subXs = tuple(self.X[i][:,indexes] for i in range(self.num_tensors))
        results = self.get(histogram, *subXs)
        return results
    
    def add(self, index, *args, **kwargs):
        values = self.evaluate_binned(*args, **kwargs)
        if not isinstance(values, tuple): values = (values,)
        super().add(index, *values)

# regression metrics parent
class StratifiedMeanErrorMetric(StratifiedMetric):
    def __init__(self, fn: Callable, *args, **kwargs):
        self.fn = fn
        super().__init__(1, *args, **kwargs)
    def evaluate_binned(self, binned_diff):
        result = reduce_binned(binned_diff, reduce_fn=self.evaluate)
        return result
    def evaluate(self, diff):
        return np.nanmean(self.fn(diff), axis=0)
    def agg(self, histogram, *arrs):
        axes = (1,2)
        assert len(arrs)==1
        return weighted_avg(arrs[0], histogram, axis=axes)
    
# regression metrics
class StratifiedMSE(StratifiedMeanErrorMetric): 
    def __init__(self, *args, **kwargs): 
        super().__init__(fn=lambda x: x**2, *args, **kwargs)
class StratifiedRMSE(StratifiedMSE):
    def evaluate(self, binned_diff):
        return np.sqrt(super().evaluate(binned_diff))    
class StratifiedMAE(StratifiedMeanErrorMetric):
    def __init__(self, *args, **kwargs): 
        super().__init__(fn=lambda x: np.abs(x), *args, **kwargs)
class StratifiedMBE(StratifiedMeanErrorMetric):
    def __init__(self, *args, **kwargs): 
        super().__init__(fn=lambda x: x, *args, **kwargs)
    
class StratifiedNLL(StratifiedMetric):
    def __init__(self, eps, *args, **kwargs):
        super().__init__(1, *args, **kwargs)
        self.eps = eps
    def evaluate_binned(self, binned_diff, binned_variance):
        result = np.full((self.num_variables, self.num_bins), np.nan)
        for i in range(self.num_variables):
            for j in range(self.num_bins):
                if binned_diff[i][j].shape==(0,) and binned_variance[i][j].shape==(0,): continue
                result[i,j] = self.evaluate(binned_diff[i][j], binned_variance[i][j])
        return result
    def evaluate(self, diff, variance):
        variance[variance<self.eps] = self.eps
        return np.nanmean(0.5 * (np.log(variance) + (diff**2)/variance), axis=0)
    def agg(self, histogram, *arrs):
        axes = (1,2)
        assert len(arrs)==1
        return weighted_avg(arrs[0], histogram, axis=axes)

# calibration metrics
class StratifiedUCE(StratifiedMetric):
    """
    X[0]: mean_variance, X[1]: mean_mse
    """
    def __init__(self, *args, **kwargs):
        super().__init__(2, *args, **kwargs)
    def evaluate_binned(self, binned_diff, binned_variance):
        return reduce_binned(binned_variance, self.evaluate, "variance"), reduce_binned(binned_diff, self.evaluate, "diff")
    def evaluate(self, values, variable):
        if variable=="variance": return np.nanmean(values, axis=0)
        elif variable=="diff": return np.nanmean(values**2, axis=0)
        else: raise AttributeError(f"`variable` must be in ['variance', 'diff']. got '{variable}'")
    def agg(self, histogram,*arrs):
        assert len(arrs)==2
        result = np.abs(np.nansum(histogram*(arrs[0]-arrs[1]), axis=self.groups_axis, keepdims=True))
        result = np.nansum(result, axis=self.bins_axis, keepdims=True)/np.nansum(histogram, axis=(self.groups_axis,self.bins_axis), keepdims=True)
        result = result.reshape(self.num_variables,-1)
        if result.shape[-1]==1: result.reshape(self.num_variables)
        return result

class StratifiedENCE(StratifiedUCE):
    """
    X[0]: mean_std, X[1]: mean_rmse
    """
    def evaluate(self, *args, **kwargs):
        return np.sqrt(super().evaluate(*args, **kwargs))
    def agg(self, histogram, *arrs):
        assert len(arrs)==2
        result = np.sqrt(np.nansum(histogram*arrs[0], axis=self.groups_axis, keepdims=True))
        result -= np.sqrt(np.nansum(histogram*arrs[1], axis=self.groups_axis, keepdims=True))
        result = np.abs(result)/np.sqrt(np.nansum(histogram*arrs[0], axis=self.groups_axis, keepdims=True))
        result = np.nansum(result, axis=self.bins_axis, keepdims=True)/np.nansum(histogram, axis=(self.groups_axis,self.bins_axis), keepdims=True)
        result = result.reshape(self.num_variables,-1)
        if result.shape[-1]==1: result.reshape(self.num_variables)
        return result

class StratifiedCIAccuracy(StratifiedMetric):
    def __init__(self, rhos, *args, **kwargs):
        super().__init__(1, *args, **kwargs)
        self.rhos = rhos
        self.num_rhos = len(rhos)
        if self.num_rhos > 1:
            self.X = np.nan*np.ones((self.num_tensors, self.num_variables, self.num_groups, self.num_bins, self.num_rhos), dtype=self.dtype)
    def evaluate_binned(self, binned_diff, binned_variance):
        # evaluate empirical acc for each variable, each bin and each confidence level
        if self.num_rhos > 1: x = np.full((self.num_variables, self.num_bins, self.num_rhos), np.nan)
        else: x = np.full((self.num_variables, self.num_bins), np.nan)
        for i in range(self.num_variables):
            for j in range(self.num_bins):
                diff = binned_diff[i][j]
                var = binned_variance[i][j]
                x[i,j] = self.evaluate(diff, var)                    
        if self.num_rhos > 1: return x
        else: return x.reshape((self.num_variables, self.num_bins))
    def evaluate(self, diff, var):
        x = np.full((self.num_rhos), np.nan)
        if len(diff)>0:
            for k, rho in enumerate(self.rhos):
                half_width = scipy.stats.norm.ppf((1+rho)/2)*np.sqrt(var)
                acc = np.count_nonzero(np.abs(diff)<half_width, axis=0)/len(diff)
                x[k] = acc
        return x
    def agg(self, histogram, *arrs):
        assert len(arrs)==1
        histogram = histogram[(...,*([np.newaxis]*(len(arrs[0].shape)-len(histogram.shape))))]
        axes = (1,2)
        return weighted_avg(arrs[0], histogram, axis=axes)
    
class StratifiedAUCE(StratifiedCIAccuracy):
    def __init__(self, lo_rho, hi_rho, num_rhos, *args, **kwargs):
        rhos = np.linspace(lo_rho, hi_rho, num_rhos)
        super().__init__(rhos, *args, **kwargs)
    def agg(self, histogram, *arrs):
        empirical = super().agg(histogram, *arrs)
        cerr = np.abs(self.rhos-empirical)
        return np.nansum(cerr[...,1:]+cerr[...,:-1], axis=-1)/(2*self.num_rhos)
    def agg_tensor(self, H, bins):
        X = self.X[...,bins,:]
        H = np.expand_dims(H, axis=-1)
        result = weighted_avg(X, H, axis=-2, keepdims=True)
        return result

# Usefulness metrics
class StratifiedCv(StratifiedMetric):
    """
    X[0]: mean_std (mu), X[1]: var_std (sum((sigma-mu)^2))
    """
    def __init__(self, *args, **kwargs):
        super().__init__(2, *args, **kwargs)
    def evaluate_binned(self, binned_variance):
        mean_stds = np.full((self.num_variables, self.num_bins), np.nan)
        var_stds = np.full((self.num_variables, self.num_bins), np.nan)
        for i in range(self.num_variables):
            for j in range(self.num_bins):
                if binned_variance[i][j].shape!=(0,):
                    mean_stds[i,j], var_stds[i,j] = self.evaluate(binned_variance[i][j])
        return mean_stds, var_stds
    def evaluate(self, var):
        std = np.sqrt(var)
        mean_std = np.nanmean(std, axis=0)
        var_std = np.nansum((std-mean_std)**2, axis=0)
        return mean_std, var_std
    def agg(self, histogram, *arrs):
        assert len(arrs)==2
        axes = (1,2)
        mu = weighted_avg(arrs[0], histogram, axis=axes, keepdims=True)
        result = np.sqrt(np.nansum(arrs[1], axis=axes, keepdims=True)/(np.nansum(histogram, axis=axes, keepdims=True)-1))/mu
        return result.reshape(self.num_variables,)
    
class StratifiedSRP(StratifiedMetric):
    def __init__(self, *args, **kwargs):
        super().__init__(1, *args, **kwargs)
    def evaluate_binned(self, binned_variance):
        return reduce_binned(binned_variance, reduce_fn=self.evaluate)
    def evaluate(self, variance):
        return np.nanmean(variance, axis=0)
    def agg(self, histogram, *arrs):
        assert len(arrs)==1
        axes = (1,2)
        return weighted_avg(arrs[0], histogram, axis=axes)
    
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
        # NLL
        eps_nll: float=1e-6,
        # AUCE
        lo_rho: float=1e-3,
        hi_rho: float=1-1e-3,
        num_rhos: int=100,
        verbose: bool=False,
    ):
        self.num_variables = num_variables
        shape_kw={"num_variables": num_variables, "num_groups": num_groups, "num_bins": num_bins}
        # stratified tensors
        self.histogram = StratifiedHistogram(lo_variance, hi_variance, **shape_kw)
        self.mean_variance = StratifiedTensor(1, **shape_kw)
        # stratified metrics
        self.mse = StratifiedMSE(**shape_kw)
        self.rmse = StratifiedRMSE(**shape_kw)
        self.mae = StratifiedMAE(**shape_kw)
        self.mbe = StratifiedMBE(**shape_kw)
        self.nll = StratifiedNLL(eps_nll, **shape_kw)
        self.uce = StratifiedUCE(**shape_kw)
        self.ence = StratifiedENCE(**shape_kw)
        self.ci90_accs = StratifiedCIAccuracy(rhos=[0.9], **shape_kw)
        self.auce = StratifiedAUCE(lo_rho, hi_rho, num_rhos, **shape_kw)
        self.cv = StratifiedCv(**shape_kw)
        self.srp = StratifiedSRP(**shape_kw)
        # other attributes
        self.index_map = dict()
        self.counter = 0
        self.results = None
        self.verbose=verbose

    def metrics_tensors(self):
        return dict(
            mse=self.mse,
            rmse=self.rmse,
            mae=self.mae,
            mbe=self.mbe,
            nll=self.nll,
            uce=self.uce,
            ence=self.ence,
            ci90_accs=self.ci90_accs,
            auce=self.auce,
            cv=self.cv,
            srp=self.srp
        )    
    
    def kwargs(self):
        return dict(
            histogram=self.histogram,
            mean_variance=self.mean_variance,
            index_map=self.index_map,
            counter=self.counter,
        )
    
    def __eq__(self, other):
        for iterator in [self.metrics_tensors(), self.kwargs()]:
            for attr_name, attr_value in iterator.items():
                other_attr = other.__getattribute__(attr_name)
                if type(other_attr)!=type(attr_value):
                    print(attr_name, "failed for type mismatch", type(other_attr),type(attr_value))
                    return False
                if isinstance(attr_value, np.ndarray): 
                    if not arrequal(attr_value,other_attr): 
                        print(attr_name, "failed for array notequal mismatch")
                        return False
                if isinstance(attr_value, StratifiedMetric) or isinstance(attr_value, StratifiedTensor):
                    if not arrequal(attr_value.X,other_attr.X):
                        print(attr_name, "failed for stratified array notequal mismatch")
                        return False
                else:
                    if attr_value!=other_attr: 
                        return False
        return True

    def empty_copy(self, k=1):
        # retrieve constructor args
        _, num_variables, num_groups, num_bins = self.histogram.shape
        assert num_bins % k == 0, f"resampling factor must give an integer, i.e. `num_bins/k` must be an integer got: {num_bins}/{k}={num_bins/k}"
        num_bins //= k
        lo_variance, hi_variance = self.histogram.lo, self.histogram.hi
        eps_nll = self.nll.eps
        lo_rho, hi_rho, num_rhos = self.auce.rhos.min(), self.auce.rhos.max(), len(self.auce.rhos)
        rcu = self.__class__(
            num_variables, num_groups, num_bins,
            lo_variance, hi_variance,
            eps_nll,
            lo_rho, hi_rho, num_rhos
        )
        return rcu
    
    def copy(self):
        rcu = self.empty_copy(k=1)
        for key, value in self.kwargs().items():
            rcu.__setattr__(key, deepcopy(value))
        for key, value in self.metrics_tensors().items():
            rcu.__setattr__(key, deepcopy(value))
        if isinstance(self.results, pd.DataFrame):
            rcu.results = self.results.copy()
        return rcu

    def save_json(self, path):
        def type_serialize(x):
            if isinstance(x, np.ndarray): return (x.tolist(), "array")
            elif isinstance(x, StratifiedTensor): return (x.X.tolist(), "array") 
            else: return (x, None)
        _, num_variables, num_groups, num_bins = self.histogram.shape
        lo_variance, hi_variance = self.histogram.lo, self.histogram.hi
        eps_nll = self.nll.eps
        lo_rho, hi_rho, num_rhos = self.auce.rhos.min(), self.auce.rhos.max(), len(self.auce.rhos)
        data = {
            "attributes": {
                "num_variables": num_variables,
                "num_groups": num_groups,
                "num_bins": num_bins,
                "lo_variance": lo_variance,
                "hi_variance": hi_variance,
                "eps_nll": eps_nll,
                "lo_rho": lo_rho,
                "hi_rho": hi_rho,
                "num_rhos": num_rhos,
                "verbose": self.verbose
            },
            "metrics_tensors": {
                metric_name: metric_tensor
                for metric_name, metric_tensor in self.metrics_tensors().items()
            },
            "kwargs": {
                kwarg_name: kwarg_value
                for kwarg_name, kwarg_value in self.kwargs().items()
            },
            "results": self.results.to_dict(orient="list") if isinstance(self.results, pd.DataFrame) else None
        }
        for upper_key in data.keys():
            if upper_key != "results":
                for lower_key, lower_value in data[upper_key].items():
                    data[upper_key][lower_key] = type_serialize(lower_value)
        with open(path, mode="w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)

    @classmethod 
    def from_json(cls, path):
        def type_deserialize(x):
            x, tp = x
            if tp == "array": return np.array(x)
            else: return x
        with open(path, mode="r", encoding="utf-8") as f:
            data = json.load(f)
            for upper_key in data.keys():
                if upper_key == "results": data[upper_key] = pd.DataFrame.from_dict(data["results"])
                else:
                    for lower_key, lower_value in data[upper_key].items():
                        data[upper_key][lower_key] = type_deserialize(lower_value)
        obj = cls(**data["attributes"])
        for metric_name, metric_tensor in data["metrics_tensors"].items():
            obj.metrics_tensors()[metric_name].__setattr__("X", np.array(metric_tensor))
        for kwarg_name, kwarg_value in data["kwargs"].items():
            if isinstance(kwarg_value, list): kwarg_value = np.array(kwarg_value)
            if isinstance(obj.kwargs()[kwarg_name], StratifiedTensor):
                obj.kwargs()[kwarg_name].__setattr__("X", kwarg_value)
            else: obj.__setattr__(kwarg_name, kwarg_value)
        if "results" in data.keys(): obj.results = data["results"]
        return obj

    def upsample(self, k:int):
        # Create empty copy with num_bins = self.num_bins//k
        rcu = self.empty_copy(k=k)
        # shapes
        _, num_variables, num_groups, num_bins = rcu.histogram.shape
        _, _, _, hi_num_bins = self.histogram.shape
        assert num_bins>1, f"upsampled StratifiedRCU requires more than 1 bin. Decrease k"
        # bin map
        bin_map = [np.arange(i, i+k) for i in np.arange(0, hi_num_bins, k)]
        # upsample histogram
        for i, bins in enumerate(bin_map):
            hi_histogram = self.histogram[...,bins]
            rcu.histogram.X[...,i] = np.nansum(hi_histogram, axis=-1)
            rcu.mean_variance.X[...,i] = weighted_avg(self.mean_variance[...,bins], hi_histogram, axis=-1)
        # upsample metrics
        for metric_name, metric_tensor in rcu.metrics_tensors().items():
            hi_metric_tensor = self.metrics_tensors()[metric_name]
            for i, bins in enumerate(bin_map):
                metric_tensor.X[:,:,:,[i]] = hi_metric_tensor.agg_tensor(self.histogram[...,bins], bins)
        return rcu

    def add_project(self, project_id: str, gt:np.ndarray, mean:np.ndarray, variance:np.ndarray):
        # nan mask flattening
        mask = ~np.isnan(mean).all(0)
        diff = mean[:,mask]-gt[:,mask]
        variance = variance[:,mask]
        # index
        self.index_map[project_id] = self.counter
        index = self.index_map[project_id]
        # add histogram
        p_variance, binned_variance, [binned_diff] = self.histogram.add(index, variance, others=[diff])
        # add mean variance
        self.mean_variance.add(index, p_variance)
        # add metrics
        for metric_name, metric in self.metrics_tensors().items():
            if self.verbose: print(f"[rcu] adding {metric.__class__.__name__} for {project_id}")
            # get metric kwargs
            kwargs = {
                k:v for k,v in list(self.kwargs().items())+[("binned_variance", binned_variance),("binned_diff", binned_diff),("counts", self.histogram[:,:,index,:])]
                if k in filter(lambda x: x!="self", inspect.getfullargspec(metric.evaluate_binned).args)
            }
            # add metric
            metric.add(index, **kwargs)
        self.counter += 1
    
    def get(self, metric_names: Optional[List[str]]=None):
        """eg rcu.get(["mse", "mbe"])"""
        # get metrics
        if metric_names is None: metric_names = list(self.metrics_tensors().keys())
        metrics = [self.metrics_tensors()[mid] for mid in metric_names]
        # compute
        results = {}
        for metric, metric_name in zip(metrics, metric_names):
            if self.verbose: print(f"[rcu] getting {metric.__class__.__name__} globally")
            res = metric.get(self.histogram)
            for k, v in res.items():
                res[k] = v.reshape(self.num_variables)
            results[metric_name] = res
        return results

    def get_subset(self, project_ids: Optional[List[str]]=None, metric_names: Optional[List[str]]=None):
        """eg rcu.get_subset(EAST, ["mse", "uce"])"""
        assert project_ids is None or isinstance(project_ids, list)
        assert metric_names is None or isinstance(metric_names, list)
        if metric_names is None: metric_names = list(self.metrics_tensors().keys())
        # get indexes and metrics
        indexes = self.index_map.values() if project_ids is None else [self.index_map[pid] for pid in project_ids]
        metrics = [self.metrics_tensors()[mid] for mid in metric_names]
        # compute
        results = {}
        for metric, metric_name in zip(metrics, metric_names):
            if self.verbose: print(f"[rcu] getting {metric.__class__.__name__} for {project_ids}")
            res =  metric.get_subset(self.histogram, indexes)
            for k, v in res.items():
                res[k] = v.reshape(self.num_variables)
            results[metric_name] = res
        return results

    def get_results_df(self, groups: Dict[str, List[str]], variable_names: List[str]):
        """
        Columns:
        - group:    [project_id], [group_id], global
        - metric:   mse, ..., srp
        - kind:     agg, ause
        - variable: p95, ..., cover
        - x:        value 
        """
        def add_result_dict(res, out, group):
            # metric loop
            for metric, mres in res.items():
                # agg/ause loop
                for kind, key in zip(["agg", "ause"], ["values", "ause"]):
                    # numbers loop
                    for i, x in enumerate(mres[key]):
                        out["group"].append(group)
                        out["metric"].append(metric)
                        out["kind"].append(kind)
                        out["variable"].append(variable_names[i])
                        out["x"].append(x)
            return out
        results = {"group": [], "metric": [], "kind": [], "variable": [], "x": []}
        for project_id in self.index_map.keys():
            res = self.get_subset([project_id])
            results = add_result_dict(res, results, project_id)
        for group_name, group_ids in groups.items():
            group_ids = list(filter(lambda x: x in self.index_map.keys(), group_ids))
            res = self.get_subset(group_ids)
            results = add_result_dict(res, results, group_name)
        res = self.get()
        results = add_result_dict(res, results, "global")
        self.results = pd.DataFrame(results)

    def plot_results(self):
        if not isinstance(self.results, pd.DataFrame):
            raise TypeError(f"`StratifiedRCU.results` is None. `StratifiedRCU.get_results_df` must be called before plotting")
        groups = ["global"] + list(filter(
            lambda x: not x in self.index_map.keys() and x != "global",
            self.results.group.unique()
        ))
        for variable in self.results.variable.unique():
            df = self.results[self.results.variable==variable]
            nrows = ceil(len(df.metric.unique()))
            ncols = ceil(len(df.metric.unique())/nrows)
            fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(12,25))
            fig.suptitle(variable)
            axs = axs.flatten()
            for i, metric in enumerate(df.metric.unique()):
                tmp = df[df.metric==metric]
                tmp2 = tmp[tmp.group.isin(self.index_map.keys())]
                tmp2 = tmp2.query(f"kind == 'agg'")
                order = groups+list(tmp2.sort_values(by=["x"]).group)
                sns.barplot(data=tmp, y="x", x="group", hue="kind", errorbar=None, ax=axs[i], order=order)
                for tick in axs[i].get_xticklabels():
                    tick.set_rotation(45)
                axs[i].set(xlabel="", ylabel="value")
                axs[i].set_title(metric.upper())
                axs[i].legend(loc='upper right')
            plt.tight_layout()
            fig.show()

    def get_calibration_curve(self, metric: str):
        metric = metric.lower()
        assert metric in ["uce", "ence", "auce"], f"{metric} is not suitable to extract a calibration curve"
        if metric in ["uce", "ence"]:
            # x is X[0], y is X[1]
            x = self.metrics_tensors()[metric].X[0] # (d,P,M)
            y = self.metrics_tensors()[metric].X[1] # (d,P,M)
            h = self.histogram.X[0]
            xc = weighted_avg(x, h, axis=1) # (d,M)
            yc = weighted_avg(y, h, axis=1) # (d,M)
        else:
            empirical_accs = self.auce.X[0]
            yc = weighted_avg(
                empirical_accs, 
                np.expand_dims(self.histogram.X[0], axis=-1), 
                axis=(1,2)
            )
            xc = self.auce.rhos
        return xc, yc

    def plot_calibration_curves(self, ks: List[int], variable_names: List[str], metrics: List[str]=["uce", "ence", "auce"]):
        assert len(variable_names)==self.num_variables
        if 1 not in ks: ks = [1]+ks
        # histograms
        ncols = self.num_variables//2
        nrows = self.num_variables-ncols
        fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(12,10))
        axs = axs.flatten()
        H = np.nansum(self.histogram.X[0], axis=1)
        for d in range(self.num_variables):
            x = np.linspace(self.histogram.lo[d], self.histogram.hi[d], self.histogram.num_bins)
            axs[d].plot(x, H[d])#, color="C0")
            axs[d].set_title(variable_names[d])
        fig.show()
        # metrics
        for metric in metrics:
            # compute calibration curves
            if metric.lower()=="auce": 
                xc, yc = self.get_calibration_curve(metric)
            else:
                xcks, ycks = [], []
                for j, k in enumerate(ks):
                    if k==1: obj = self.copy()
                    else: obj = self.upsample(k)
                    xc, yc = obj.get_calibration_curve(metric)
                    xcks.append(xc)
                    ycks.append(yc)
            # create plot objects
            if metric.lower()=="auce":
                fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12,5))
                fig.suptitle(metric.upper())
                plt.plot(np.linspace(0,1,2), np.linspace(0,1,2), color="black", linestyle="dotted")
                for d in range(self.num_variables):
                    var = variable_names[d]
                    ax.plot(xc,yc[d],label=var)
                ax.set(xlabel="expected accuracy", ylabel="empirical accuracy")
                ax.legend(loc='upper left')
            else:
                ncols = self.num_variables//2
                nrows = self.num_variables-ncols
                fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(12,10))
                axs = axs.flatten()
                fig.suptitle(metric.upper())
                xc1, yc1 = self.get_calibration_curve(metric)
                for d in range(self.num_variables):
                    axs[d].set_title(variable_names[d])
                    if metric.lower() == "ence":
                        axs[d].set(xlabel="bin std", 
                                ylabel="bin rmse")
                    else:
                        axs[d].set(xlabel="bin variance", 
                                ylabel="bin mse")
                    # plot calibration curves
                    id_line = (np.linspace(np.nanmin(xc1[d]), np.nanmax(xc1[d]), 2), np.linspace(np.nanmin(xc1[d]), np.nanmax(xc1[d]), 2))
                    axs[d].plot(*id_line, color="black", linestyle="dotted")
                    for j, k in enumerate(ks):
                        if k==1:
                            axs[d].scatter(xcks[j][d], ycks[j][d], label=f"k={k}", alpha=.2, color=f"C{j}", edgecolor=None, marker=".")
                        else:
                            axs[d].plot(xcks[j][d], ycks[j][d], label=f"k={k}", color=f"C{j}")
                    axs[d].legend(loc='upper left')
            plt.tight_layout()
            fig.show()
    
    def plot_residuals(self, groups: Dict[str, List[str]], variable_names: List[str]):
        """
        for each group and globally:
        - residuals histogram
        - residuals qqplot
        
        residuals are estimated by the mean bias error in each bins (self.mbe)
        """
        import statsmodels.api as sm
        from scipy import stats
        import matplotlib.cm as cm
        assert len(variable_names)==self.num_variables
        # add global group
        groups["global"] = list(self.index_map.keys())
        # one plot per variable with all groups
        nrows = 3
        ncols = ceil(self.num_variables/nrows)
        hfig, haxs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(12,10))
        haxs = haxs.flatten()
        qqfig, qqaxs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(12,10))
        qqaxs = qqaxs.flatten()
        # collect residuals
        residuals = {}
        pvalues = {}
        for i, var in enumerate(variable_names):
            var_residuals = {"values": [], "group": []}
            pvalues[var] = {}
            colors = iter(cm.rainbow(np.linspace(0, 1, len(groups.keys()))))
            for (group_name, group_ids) in groups.items():
                group_ids = [self.index_map[gidx] for gidx in group_ids if gidx in self.index_map.keys()]
                if len(group_ids)>0:
                    # select residuals and histogram
                    mbe = self.mbe.X[:,i,group_ids]
                    h = self.histogram.X[:,i,group_ids]
                    # aggregate across projects
                    mbe = weighted_avg(mbe, h, axis=1).squeeze(0)
                    # gaussian fit
                    norm = stats.norm(loc=np.nanmean(mbe), scale=np.nanstd(mbe)).rvs(size=mbe.shape[0])
                    pvalues[var][group_name] = stats.kstest(mbe, cdf=norm).pvalue
                    # add residuals
                    var_residuals["values"].extend(mbe.tolist())
                    var_residuals["group"].extend([f"{group_name} (p={pvalues[var][group_name]:.1e})".format() for _ in mbe.tolist()])
                    color = next(colors)
                    mbe_std = (mbe-np.nanmean(mbe))/np.nanstd(mbe)
                    sm.qqplot(mbe_std, ax=qqaxs[i], line="45", label=group_name, markerfacecolor=color, markeredgecolor=color, alpha=0.2)
                    # stats.probplot(mbe, plot=qqaxs[i])
            df = pd.DataFrame(var_residuals)
            sns.histplot(data=df, x="values", hue="group", multiple="dodge", ax=haxs[i], bins=15)
            haxs[i].set_title(f"{var}")# (pvalues: {', '.join([f'{group}={pval:.1e}' for group, pval in pvalues[var].items()])})")
            qqaxs[i].set_title(var)
            qqaxs[i].legend(loc='lower right')
            residuals[var] = var_residuals
        plt.tight_layout()
        hfig.show()
        qqfig.show()

class StratifiedRCUSubset(StratifiedRCU):
    def __init__(self, metric_names, *args, **kwargs):
        self.metric_names = metric_names
        super().__init__(*args, **kwargs)
    def metrics_tensors(self):
        return {k:v for k,v in super().metrics_tensors().items() if k in self.metric_names}
