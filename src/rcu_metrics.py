import numpy as np
import inspect
import scipy.stats
from itertools import chain
from typing import *

# Utils
def weighted_avg(values, counts, axis):
    return np.nansum(values*counts, axis=axis)/np.nansum(counts, axis=axis)

def apply_binned(binned, fn, *args):
    num_variables, num_bins = len(binned), len(binned[0])
    result = [[[] for _ in range(num_bins)] for _ in range(num_variables)]
    for i in range(num_variables):
        for j in range(num_bins):
            for k in range(len(binned[i][j])):
                result[i][j].append(fn(binned[i][j][k], *args)) # apply fn to binned[i][j][k]
    return result

def reduce_binned(binned, reduce_fn, *args):
    num_variables, num_bins = len(binned), len(binned[0])
    result = np.full((num_variables, num_bins), np.nan)
    for i in range(num_variables):
        for j in range(num_bins):
            result[i,j] = reduce_fn(binned[i][j], *args) # reduce list binned[i][j] into single value
    return result

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
        print("ST init:", self.num_variables, self.num_groups, self.num_bins)
        self.X = np.nan*np.ones((num_variables, num_groups, num_bins), dtype=dtype)
        self.variables_axis, self.groups_axis, self.bins_axis = 0, 1, 2
    @property
    def array(self): return self.X
    @property
    def dtype(self): return self.X.dtype    
    def __getattribute__(self, attr: str):
        try: self.X.__getattribute__(attr)
        except: super().__getattribute__(attr)    
    def add(self, index: int, values: np.ndarray):
        """assign along group axis"""
        self.X[:,index] = values

class DualStratifiedTensor:
    def __init__(
        self,
        num_variables: int, 
        num_groups: int, 
        num_bins: int,
        dtype: Union[str, np.dtype]=np.float32
    ):
        """
        (d, P, M, *), (d, P, M, *)
        """
        self.num_variables, self.num_groups, self.num_bins = num_variables, num_groups, num_bins
        self.X1 = np.nan*np.ones((num_variables, num_groups, num_bins), dtype=dtype)
        self.X2 = np.nan*np.ones((num_variables, num_groups, num_bins), dtype=dtype)
        self.variables_axis, self.groups_axis, self.bins_axis = 0, 1, 2
    @property
    def array(self): return (self.X1, self.X2)
    @property
    def dtype(self): return self.X1.dtype    
    def __getattribute__(self, attr: str):
        try: (self.X1.__getattribute__(self, attr), self.X2.__getattribute__(self, attr))
        except: super().__getattribute__(attr)    
    def add(self, index: int, values1: np.ndarray, values2: np.ndarray):
        """assign along group axis"""
        self.X1[:,index] = values1
        self.X2[:,index] = values2

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
        print("STH init:", args, kwargs)
        super().__init__(*args, **kwargs)
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
                self.X[:,index,bin_id] = mask.sum(axis=1)
                binned_values[i][bin_id] = values[i,variable_mask]
                for j, other in enumerate(others): 
                    binned_others[j][i][bin_id] = others[j][i,variable_mask]
        # compute mean values
        mean_values = self.reduce_binned(binned_values, reduce_fn=lambda x: np.nanmean(x, axis=0))
        return mean_values, binned_values, binned_others

# metrics Parent
class StratifiedMetric(StratifiedTensor):
    """
    Idea: compute at the highest resolution (i.e. for all bin and all variables) and then aggregate on demand
    """
    def evaluate_binned(self, *args, **kwargs):
        """compute the metric in each bin and each variables"""
        raise NotImplementedError()

    def evaluate(self, *args, **kwargs):
        """actually compute the metric in a given set of variables"""
        raise NotImplementedError()

    def agg(self, histogram, arr=None, axis=None, keepbins=False): 
        """aggregate (d,P,M)->(d,) if not keepbins else (d,P,M)->(d,M)"""
        raise NotImplementedError()
    
    def cumagg(self, histogram, arr=None):
        """cumulatively aggregate"""
        if arr is None: arr = self.X
        cumX = np.nan*np.ones((self.num_variables, self.num_bins))
        cumH = np.nancumsum(histogram, axis=self.group_axis)
        for k in range(self.num_bins):
            cumX[:,k] = self.agg(cumH[:,:,:k], arr[:,:,:k], keepbins=True)
        return cumX
    
    def ause(self, histogram, arr=None): 
        cumX = self.cumagg(histogram, arr)
        return np.nansum((cumX[:,1:]+cumX[:,:-1]), axis=self.bins_axis-1)/(2*self.num_bins)

    def get(self, histogram, arr=None):
        results = {"values": None, "ause": None}
        if arr is None: arr = self.X
        results["values"] = self.agg(histogram, arr)
        results["ause"] = self.ause(histogram, arr)
        return results
    
    def get_subset(self, histogram, indexes):
        subX = self.X[indexes,:]
        return self.get(histogram, subX)
    
# dual metrics Parent
class DualStratifiedMetric(DualStratifiedTensor):
    def agg(self, histogram, arr=None, axis=None, keepbins=False): 
        """aggregate (d,P,M)->(d,) if not keepbins else (d,P,M)->(d,M)"""
        raise NotImplementedError()
    
    def cumagg(self, histogram, arr1, arr2, cumaxis=None, aggaxis=None):
        if arr1 is None: arr1 = self.X1
        if arr2 is None: arr2 = self.X2
        cumX = np.nan*np.ones((self.num_variables, self.num_bins))
        cumH = np.nancumsum(histogram, axis=self.group_axis)
        for k in range(self.num_bins):
            cumX[:,k] = self.agg_(cumH[:,:,:k], arr1[:,:,:k], arr2[:,:,:k], keepbins=True)
        return cumX

    def ause(self, histogram, arr1=None, arr2=None): 
        cumX = self.cumagg(histogram, arr1, arr2)
        return np.nansum((cumX[:,1:]+cumX[:,:-1]), axis=self.bins_axis-1)/(2*self.num_bins)

    def get(self, histogram, arr1=None, arr2=None):
        results = {"values": None, "ause": None}
        if arr1 is None: arr1 = self.X1
        if arr2 is None: arr2 = self.X2
        results["values"] = self.agg(histogram, arr1, arr2)
        results["ause"] = self.ause(histogram, arr1, arr2)
        return results
    
    def get_subset(self, histogram, indexes):
        subX1 = self.X1[indexes,:]
        subX2 = self.X2[indexes,:]
        return self.get(histogram, subX1, subX2)

# regression metrics parent
class StratifiedMeanErrorMetric(StratifiedMetric):
    def __init__(self, fn: Callable, *args, **kwargs):
        self.fn = fn
        super().__init__(*args, **kwargs)
    def evaluate_binned(self, binned_diff):
        return reduce_binned(binned_diff, reduce_fn=self.evaluate)
    def evaluate(self, diff):
        return self.nanmean(self.fn(diff), axis=0)
    def agg(self, histogram, arr=None, keepbins=False):
        if arr is None: arr = self.X
        axes = 1 if keepbins else (1,2)
        return weighted_avg(arr, histogram, axis=axes)
    
# regression metrics
class StratifiedMSE(StratifiedMeanErrorMetric): 
    def __init__(self, *args, **kwargs): 
        super().__init__(fn=lambda x: x**2)
class StratifiedRMSE(StratifiedMSE):
    def evaluate(self, binned_diff):
        return np.sqrt(super().evaluate(binned_diff))    
class StratifiedMAE(StratifiedMeanErrorMetric):
    def __init__(self, *args, **kwargs): 
        super().__init__(fn=lambda x: np.abs(x))
class StratifiedMBE(StratifiedMeanErrorMetric):
    def __init__(self, *args, **kwargs): 
        super().__init__(fn=lambda x: x)
    
class StratifiedNLL(StratifiedMetric):
    def __init__(self, eps, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.eps = eps
    def evaluate_binned(self, binned_diff, binned_variance):
        result = np.full((self.num_variables, self.num_bins), np.nan)
        for i in range(self.num_variables):
            for j in range(self.num_bins):
                result[i,j] = self.evaluate(binned_diff[i][j], binned_variance[i][j])
        return result
    def evaluate(self, diff, variance):
        variance[variance<self.eps] = self.eps
        return np.nanmean(0.5 * (np.log(variance) + (diff**2)/variance), axis=0)
    def agg(self, histogram, arr=None, keepbins=False):
        if arr is None: arr = self.X
        axes = 1 if keepbins else (1,2)
        return weighted_avg(arr, histogram, axis=axes)

# calibration metrics
class StratifiedUCE(DualStratifiedMetric):
    """
    X1: mean_variance, X2: mean_mse
    """
    def evaluate_binned(self, binned_diff, binned_variance):
        return reduce_binned(binned_variance, self.evaluate, "variance"), reduce_binned(binned_diff, self.evaluate, "diff")
    def evaluate(self, values, variable):
        if variable=="variance": return np.nanmean(values, axis=0)
        elif variable=="diff": return np.nanmean(values**2, axis=0)
        else: raise AttributeError(f"`variable` must be in ['variance', 'diff']. got '{variable}'")
    def agg(self, histogram, arr1=None, arr2=None, keepbins=False):
        if arr1 is None: arr1 = self.X1
        if arr2 is None: arr2 = self.X2
        N = histogram.sum(axis=(1,2)) if not keepbins else histogram.sum(axis=1)
        result = self.X1-self.X2
        result = np.abs(np.nansum(result*histogram, axis=self.groups_axis, keepdims=True))
        if keepbins:
            result = result.reshape((self.num_variables,self.num_bins))
        else:
            result = np.nansum(result, axis=self.bins_axis, keepdims=True)
            result = result.reshape((self.num_variables,))
        result /= N
        return result

class StratifiedENCE(StratifiedUCE):
    """
    X1: mean_variance, X2: mean_mse
    """
    def evaluate(self, *args, **kwargs):
        return np.sqrt(super().evaluate(*args, **kwargs))
    def agg(self, histogram, arr1=None, arr2=None, keepbins=False):
        if arr1 is None: arr1 = self.X1
        if arr2 is None: arr2 = self.X2
        if axis is None: axis = self.groups_axis
        N = histogram.sum(axis=1) if not keepbins else histogram.sum(axis=(1,2))
        r1 = np.sqrt(np.nansum(histogram*self.X1, axis=self.groups_axis, keepdims=True))
        r2 = np.sqrt(np.nansum(histogram*self.X2, axis=self.groups_axis, keepdims=True))
        result = r1-r2
        result = np.abs(result) / r1
        if keepbins:
            result = result.reshape((self.num_variables,self.num_bins))
        else:
            result = np.nansum(result, axis=self.bins_axis, keepdims=True)
            result = result.reshape((self.num_variables,))
        result /= N
        return result

class StratifiedCIAccuracy(StratifiedMetric):
    def __init__(self, rhos, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rhos = rhos
        self.num_rhos = len(rhos)
        self.X = np.nan*np.ones((self.num_variables, self.num_groups, self.num_bins, self.num_rhos), dtype=self.dtype)
    def evaluate_binned(self, binned_diff, binned_variance):
        # evaluate empirical acc for each variable, each bin and each confidence level
        x = np.full((self.num_variables, self.num_bins, self.num_rhos), np.nan)
        for i in range(self.num_variables):
            for j in range(self.num_bins):
                diff = binned_diff[i,j]
                var = binned_variance[i,j]
                x[i,j,:] = self.evaluate(diff, var)                    
        return x
    def evaluate(self, diff, var):
        x = np.full((self.num_rhos), np.nan)
        if len(diff)>0:
            for k, rho in enumerate(self.rhos):
                half_width = scipy.stats.norm.ppf((1+rho)/2)*np.sqrt(var)
                acc = np.count_nonzero(np.abs(diff)<half_width, axis=0)/len(diff)
                x[k] = acc
        return x
    def agg(self, histogram, arr=None, keepbins=False):
        if arr is None: arr = self.X
        axes = 1 if keepbins else (1,2)
        return weighted_avg(arr, histogram, axis=axes)

class StratifiedAUCE(StratifiedMetric):
    def __init__(self, lo_rho, hi_rho, num_rhos, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rhos = np.linspace(lo_rho, hi_rho, num_rhos)
        self.ci_accs = StratifiedCIAccuracy(self.rhos)
    def evaluate_binned(self, binned_diff, binned_variance):
        x = np.full((self.num_variables, self.num_bins), np.nan)
        accs = np.full((self.num_variables, self.num_bins, self.ci_accs.num_rhos), np.nan)
        for i in range(self.num_variables):
            for j in range(self.num_bins):
                diff = binned_diff[i,j]
                var = binned_variance[i,j]
                auce, acc = self.evaluate(diff, var)
                x[i,j] = auce
                accs[i,j,:] = acc
        return x
    def evaluate(self, diff, var):
        if len(diff)>0:
            empirical_accs = self.ci_accs.evaluate(diff, var) # (R,)
            error = np.abs(self.rhos-empirical_accs)
            return np.nansum((error[:,1:]+error[:,:-1]), axis=0)/(2*self.ci_accs.num_rhos), empirical_accs # ((1,), (R,))
        else:
            return np.nan, np.full((self.ci_accs.num_rhos), np.nan) # ((1,), (R,))
    def agg(self, histogram, arr=None, keepbins=False):
        if arr is None: arr = self.X
        empirical_accs = self.ci_accs.agg(histogram, arr, keepbins) # (d,R) or (d,M,R)
        error = np.abs(self.rhos-empirical_accs)
        if keepbins: return np.nansum((error[:,:,1:]+error[:,:,:-1]), axis=2)/(2*self.ci_accs.num_rhos)
        else: return np.nansum((error[:,1:]+error[:,:-1]), axis=1)/(2*self.ci_accs.num_rhos)

# Usefulness metrics
class StratifiedCv(StratifiedMetric): 
    """
    X1: mean_std(bin)=mean_std, X2: C_v(bin)*mean_std(dataset)=var_std
    """
    def evaluate_binned(self, binned_variance):
        # to std
        binned_std = apply_binned(binned_variance, fn=lambda x: np.sqrt(x))
        # compute mean std
        mean_std = reduce_binned(binned_std, self.evaluate, "mean_std") #(d,M)
        # unbias std: not so elegant sorry
        binned_std = [
            [
                list(np.array(binned_std[i][j])-mean_std[i][j])
                for j in range(self.num_bins)
            ] 
            for i in range(self.num_variable)
        ]
        # compute std variance in each bin/variable
        var_std = reduce_binned(binned_std, self.evaluate, "var_std") #(d,M)
        return mean_std, var_std
    def evaluate(self, values, variable):
        if variable=="mean_std": return np.nanmean(values, axis=0)
        elif variable=="var_std": 
            return np.sqrt(np.nansum(values, axis=0)/(values.shape[0]-1))
        else: raise AttributeError(f"`variable` must be in ['mean_std', 'var_std']. got '{variable}'")
    def agg(self, histogram, arr1=None, arr2=None, keepbins=False):
        if arr1 is None: arr1 = self.X1
        if arr2 is None: arr2 = self.X2
        mu = weighted_avg(arr1, histogram, axis=1) if keepbins else weighted_avg(arr1, histogram, axis=(1,2))
        result = (histogram-1)*self.arr2
        axes = 1 if keepbins else (1,2)
        result = np.sqrt(np.nansum(result, axis=axes)/(np.nansum(histogram, axis=axes)-1))/mu
        return result
    
class StratifiedSRP(StratifiedMetric):
    def evaluate_binned(self, binned_variance):
        return reduce_binned(binned_variance, reduce_fn=self.evaluate)
    def evaluate(self, variance):
        return self.nanmean(self.fn(variance), axis=0)
    def agg(self, histogram, arr=None, keepbins=False):
        if arr is None: arr = self.X
        axes = 1 if keepbins else (1,2)
        return weighted_avg(arr, histogram, axis=axes)
    
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
    ):
        shape_kw={"num_variables": num_variables, "num_groups": num_groups, "num_bins": num_bins}
        # stratified tensors
        self.histogram = StratifiedHistogram(lo_variance, hi_variance, **shape_kw)
        self.mean_variance = StratifiedTensor(**shape_kw)
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

    @property
    def metrics_tensors(self):
        return dict(
            mse=self.mse,
            rmse=self.rmse,
            mae=self.mae,
            mbe=self.mbe,
            nll=self.nll,
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
        # add mean variance
        self.variance.add(index, p_variance)
        # compute mean diffs for mean error metrics
        error_kwargs = {}
        for metric_name, metric in [("mse", self.mse), ("rmse", self.rmse), ("mae", self.mae), ("mbe", self.mbe)]:
            values = self.histogram.reduce_binned(binned_diff, lambda x: np.nanmean(metric.fn(x)))
            error_kwargs[metric_name] = values
            metric.add(index, values)
        # add metrics
        for metric in self.metrics_tensors.values():
            # get metric kwargs
            kwargs = {
                k:v for k,v in list(self.kwargs.items())+[("binned_variance", binned_variance),("binned_diff", binned_diff),("counts", self.histogram[:,index,:])]
                if k in filter(lambda x: x!="self", inspect.getfullargspec(metric.evaluate).args)
            }
            # add metric
            values = metric.evaluate_binned(**kwargs)
            if DualStratifiedMetric in metric.__class__.__bases__: metric.add(index, values[0], values[1])
            elif metric==self.auce:
                auce_values, auce_acc_values = values
                self.auce.add(index, auce_values)
                self.auce.ci_accs.add(index, auce_acc_values)
            else: metric.add(index, values)
    
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
        """eg rcu.get_subset(EAST, ["mse", "uce"])"""
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

    def get_all(self,  metric_names: Optional[List[str]]=None):
        return self.get_subset(metric_names)
