import numpy as np
import inspect
import scipy.stats
from itertools import chain
from typing import *
from copy import deepcopy

# Utils
def arrequal(a, b):
    if np.isnan(a).sum()==a.flatten().shape[0] and np.isnan(b).sum()==b.flatten().shape[0]:
        return True # full nan arrays
    try: np.testing.assert_equal(a,b)
    except: return False
    return True

def print_modified(clsn, xc, x, attr):
    print(xc)
    print(x)
    print(f"{clsn} modified {attr}")
    print(f"{attr}c: {xc.shape}, {attr}: {x.shape}")
    print(f"sum({attr}c): {np.nansum(xc)}, sum({attr}): {np.nansum(x)}")
    for i in range(5):
        for j in range(2):
            print(f"ndiff [{i},{j}]: {attr}c={xc[i,j]}, {attr}={x[i,j]}, diff: {np.setdiff1d(xc[i,j], x[i,j])}, eq: {arrequal(xc[i,j], x[i,j])}")

def nan_frac(arr):
    return np.isnan(arr).sum()/(np.isnan(arr).sum()+(~np.isnan(arr)).sum())

def weighted_avg(values, counts, axis, keepdims=False):
    # print("[debug:9]", values.shape, counts.shape, (values*counts).shape, np.nansum(values*counts, axis=axis).shape, np.nansum(counts, axis=axis).shape)
    return np.nansum(values*counts, axis=axis, keepdims=keepdims)/np.nansum(counts, axis=axis, keepdims=keepdims)

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
            if binned[i][j].shape==(0,): continue
            result[i,j] = reduce_fn(binned[i][j], *args) # reduce list binned[i][j] into single value
    return result

# stratified tensor parent
class StratifiedTensor():
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
        self.X = np.nan*np.ones((num_variables, num_groups, num_bins), dtype=dtype)
        self.num_variables, self.num_groups, self.num_bins = self.X.shape
        self.variables_axis, self.groups_axis, self.bins_axis = 0, 1, 2
    def __getitem__(self, sel): return self.X.__getitem__(sel)
    @property
    def array(self): return self.X
    @property
    def dtype(self): return self.X.dtype    
    @property
    def shape(self): return self.X.shape
    def add(self, index: int, values: np.ndarray):
        """assign along group axis"""
        self.X[:,index] = values
    def __eq__(self, other):
        if type(other)!=type(self): return False
        else: 
            if not arrequal(self.X,other.X): return False 
        return True

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
    def __getitem__(self, sel): return self.X1.__getitem__(sel), self.X2.__getitem__(sel)
    @property
    def array(self): return (self.X1, self.X2)
    @property
    def dtype(self): return self.X1.dtype
    @property
    def shape(self): 
        assert self.X1.shape==self.X2.shape     
        return self.X2.shape
    def add(self, index: int, values1: np.ndarray, values2: np.ndarray):
        """assign along group axis"""
        self.X1[:,index] = values1
        self.X2[:,index] = values2
    def __eq__(self, other):
        if type(other)!=type(self): return False
        else: 
            if (not arrequal(self.X1,other.X1)) or (not arrequal(self.X2,other.X2)): return False
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
        mean_values = reduce_binned(binned_values, reduce_fn=lambda x: np.nanmean(x, axis=0))
        return mean_values, binned_values, binned_others

# metrics Parent
class StratifiedMetric(StratifiedTensor):
    """
    Idea: compute at the highest resolution (i.e. for all bin and all variables) and then aggregate on demand
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.computed_results = {}

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
        if not isinstance(histogram, np.ndarray): histogram = histogram.array.copy()
        if arr is None: arr = self.X.copy()
        cumshape = [self.num_variables, self.num_bins]
        if self.__class__.__name__ == "StratifiedCIAccuracy": cumshape += list(self.X.shape[3:])
        cumX = np.nan*np.ones(cumshape)
        cumH = np.nancumsum(histogram, axis=self.groups_axis)
        for k in range(1,self.num_bins):
            cumX[:,k] = self.agg(cumH[:,:,:k], arr[:,:,:k], keepbins=False).reshape(cumX[:,k].shape)
        return cumX
    
    def ause(self, histogram, arr=None): 
        if not isinstance(histogram, np.ndarray): histogram = histogram.array.copy()
        cumX = self.cumagg(histogram, arr)
        return np.nansum((cumX[:,1:]+cumX[:,:-1]), axis=self.bins_axis-1)/(2*self.num_bins)

    def get(self, histogram, arr=None):
        globalres = arr is None
        print(f"[debug:217] globalres? {globalres}")
        #if globalres is None and "__all__" in self.computed_results.keys(): return self.computed_results["__all__"]
        if not isinstance(histogram, np.ndarray): histogram = histogram.array.copy()
        xc, hc = self.X.copy(), histogram.copy()
        results = {"values": None, "ause": None}
        if arr is None: arr = self.X.copy()
        results["values"] = self.agg(histogram, arr)
        results["ause"] = self.ause(histogram, arr)
        assert arrequal(xc,self.X), print_modified(self.__class__.__name__,xc,self.X, "X")
        assert arrequal(hc,histogram), print_modified(self.__class__.__name__,hc,histogram, "H")
        if globalres: self.computed_results["__all__"] = results
        print(f"[debug:227] @get, reskeys: {self.computed_results.keys()}")
        return results
    
    def get_subset(self, histogram, indexes):
        print(f"[debug:213] getting {indexes}")
        reskey = "__"+"_".join(sorted([str(xx) for xx in indexes]))+"__"
        #if reskey in self.computed_results.keys(): return self.computed_results[reskey]
        if not isinstance(histogram, np.ndarray): histogram = histogram.array.copy()
        subX = self.X[:,indexes,:].copy()
        histogram = histogram[:,indexes,:].copy()
        results = self.get(histogram, subX)
        self.computed_results[reskey] = results
        print(f"[debug:239] @get_subset, reskeys: {self.computed_results.keys()}")
        return results
    
# dual metrics Parent
class DualStratifiedMetric(DualStratifiedTensor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.computed_results = {}

    def evaluate_binned(self, *args, **kwargs):
        """compute the metric in each bin and each variables"""
        raise NotImplementedError()

    def evaluate(self, *args, **kwargs):
        """actually compute the metric in a given set of variables"""
        raise NotImplementedError()

    def agg(self, histogram, arr=None, axis=None, keepbins=False): 
        """aggregate (d,P,M)->(d,) if not keepbins else (d,P,M)->(d,M)"""
        raise NotImplementedError()
    
    def cumagg(self, histogram, arr1=None, arr2=None):
        if not isinstance(histogram, np.ndarray): histogram = histogram.array.copy()
        if arr1 is None: arr1 = self.X1.copy()
        if arr2 is None: arr2 = self.X2.copy()
        cumX = np.nan*np.ones((self.num_variables, self.num_bins))
        cumH = np.nancumsum(histogram, axis=self.groups_axis)
        print(f"[Debug:218] cumX: {cumX.shape}, cumH: {cumH.shape}")
        for k in range(1,self.num_bins):
            cumX[:,k] = self.agg(cumH[:,:,:k], arr1[:,:,:k], arr2[:,:,:k], keepbins=False).reshape(cumX[:,k].shape)
        return cumX

    def ause(self, histogram, arr1=None, arr2=None): 
        if not isinstance(histogram, np.ndarray): histogram = histogram.array.copy()
        cumX = self.cumagg(histogram, arr1, arr2)
        return np.nansum((cumX[:,1:]+cumX[:,:-1]), axis=self.bins_axis-1)/(2*self.num_bins)

    def get(self, histogram, arr1=None, arr2=None):
        globalres = arr1 is None and arr2 is None
        print(f"[debug:280] globalres? {globalres}")
        #if globalres is None and "__all__" in self.computed_results.keys(): return self.computed_results["__all__"]
        if not isinstance(histogram, np.ndarray): histogram = histogram.array.copy()
        x1c, x2c, hc = self.X1.copy(), self.X2.copy(), histogram.copy()
        results = {"values": None, "ause": None}
        if arr1 is None: arr1 = self.X1.copy()
        if arr2 is None: arr2 = self.X2.copy()
        results["values"] = self.agg(histogram, arr1, arr2)
        results["ause"] = self.ause(histogram, arr1, arr2)
        assert arrequal(x1c,self.X1), print_modified(self.__class__.__name__,x1c,self.X1, "X1")
        assert arrequal(x2c,self.X2), print_modified(self.__class__.__name__,x2c,self.X2, "X2")
        assert arrequal(hc,histogram), print_modified(self.__class__.__name__,hc,histogram, "H")
        if globalres: self.computed_results["__all__"] = results
        print(f"[debug:293] @get, reskeys: {self.computed_results.keys()}")
        return results
    
    def get_subset(self, histogram, indexes):
        reskey = "__"+"_".join(sorted([str(xx) for xx in indexes]))+"__"
        #if reskey in self.computed_results.keys(): return self.computed_results[reskey]
        if not isinstance(histogram, np.ndarray): histogram = histogram.array.copy()
        subX1 = self.X1[indexes,:].copy()
        subX2 = self.X2[indexes,:].copy()
        results = self.get(histogram, subX1, subX2)
        self.computed_results[reskey] = results
        return results

# regression metrics parent
class StratifiedMeanErrorMetric(StratifiedMetric):
    def __init__(self, fn: Callable, *args, **kwargs):
        self.fn = fn
        super().__init__(*args, **kwargs)
    def evaluate_binned(self, binned_diff):
        return reduce_binned(binned_diff, reduce_fn=self.evaluate)
    def evaluate(self, diff):
        return np.nanmean(self.fn(diff), axis=0)
    def agg(self, histogram, arr=None, keepbins=False):
        if not isinstance(histogram, np.ndarray): histogram = histogram.array.copy()
        if arr is None: arr = self.X.copy()
        axes = 1 if keepbins else (1,2)
        return weighted_avg(arr, histogram, axis=axes)
    
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
        super().__init__(*args, **kwargs)
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
    def agg(self, histogram, arr=None, keepbins=False):
        if not isinstance(histogram, np.ndarray): histogram = histogram.array.copy()
        if arr is None: arr = self.X.copy()
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
        if not isinstance(histogram, np.ndarray): histogram = histogram.array.copy()
        if arr1 is None: arr1 = self.X1.copy()
        if arr2 is None: arr2 = self.X2.copy()
        result = np.abs(np.nansum(histogram*(arr1-arr2), axis=self.groups_axis, keepdims=True))
        if keepbins:
            result = result/np.nansum(histogram, axis=self.groups_axis, keepdims=True)
            result = result.reshape(self.num_variables,self.num_bins)
        else:
            result = np.nansum(result, axis=self.bins_axis, keepdims=True)/np.nansum(histogram, axis=(self.groups_axis,self.bins_axis), keepdims=True)
            result = result.reshape(self.num_variables,)
        return result

class StratifiedENCE(StratifiedUCE):
    """
    X1: mean_variance, X2: mean_mse
    """
    def evaluate(self, *args, **kwargs):
        return np.sqrt(super().evaluate(*args, **kwargs))
    def agg(self, histogram, arr1=None, arr2=None, keepbins=False):
        if not isinstance(histogram, np.ndarray): histogram = histogram.array.copy()
        if arr1 is None: arr1 = self.X1.copy()
        if arr2 is None: arr2 = self.X2.copy()
        result = np.sqrt(np.nansum(histogram*arr1, axis=self.groups_axis, keepdims=True))
        result -= np.sqrt(np.nansum(histogram*arr2, axis=self.groups_axis, keepdims=True))
        result = np.abs(result)/np.sqrt(np.nansum(histogram*arr1, axis=self.groups_axis, keepdims=True))
        if keepbins:
            result = result / np.nansum(histogram, axis=self.groups_axis, keepdims=True)
            result = result.reshape(self.num_variables,self.num_bins)
        else:
            result = np.nansum(result, axis=self.bins_axis, keepdims=True)/np.nansum(histogram, axis=(self.groups_axis,self.bins_axis), keepdims=True)
            result = result.reshape(self.num_variables,)
        return result

class StratifiedCIAccuracy(StratifiedMetric):
    def __init__(self, rhos, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rhos = rhos
        self.num_rhos = len(rhos)
        if self.num_rhos > 1:
            self.X = np.nan*np.ones((self.num_variables, self.num_groups, self.num_bins, self.num_rhos), dtype=self.dtype)
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
    def agg(self, histogram, arr=None, keepbins=False):
        if not isinstance(histogram, np.ndarray): histogram = histogram.array.copy()
        if arr is None: arr = self.X.copy()
        histogram = histogram[(...,*([np.newaxis]*(len(arr.shape)-len(histogram.shape))))]
        axes = 1 if keepbins else (1,2)
        return weighted_avg(arr, histogram, axis=axes)
    
class StratifiedAUCE(StratifiedCIAccuracy):
    def __init__(self, lo_rho, hi_rho, num_rhos, *args, **kwargs):
        rhos = np.linspace(lo_rho, hi_rho, num_rhos)
        super().__init__(rhos, *args, **kwargs)
    def agg(self, histogram, arr=None, keepbins=False):
        if not isinstance(histogram, np.ndarray): histogram = histogram.array.copy()
        empirical = super().agg(histogram, arr, keepbins)
        cerr = np.abs(self.rhos-empirical)
        return np.nansum(cerr[...,1:]+cerr[...,:-1], axis=-1)/(2*self.num_rhos)

# Usefulness metrics
class StratifiedCv(DualStratifiedMetric):
    """
    X1: mean_std (mu), X1: var_std (sum((sigma-mu)^2))
    """
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
    def agg(self, histogram, arr1=None, arr2=None, keepbins=False):
        if not isinstance(histogram, np.ndarray): histogram = histogram.array.copy()
        if arr1 is None: arr1 = self.X1.copy()
        if arr2 is None: arr2 = self.X2.copy()
        axes = (self.groups_axis,self.bins_axis) if not keepbins else self.groups_axis
        mu = weighted_avg(arr1, histogram, axis=axes, keepdims=True)
        result = np.sqrt(np.nansum(arr2, axis=axes, keepdims=True)/(np.nansum(histogram, axis=axes, keepdims=True)-1))/mu
        if keepbins: return result.reshape(self.num_variables, self.num_bins)
        else: return result.reshape(self.num_variables,)
    
class StratifiedSRP(StratifiedMetric):
    def evaluate_binned(self, binned_variance):
        return reduce_binned(binned_variance, reduce_fn=self.evaluate)
    def evaluate(self, variance):
        return np.nanmean(variance, axis=0)
    def agg(self, histogram, arr=None, keepbins=False):
        if not isinstance(histogram, np.ndarray): histogram = histogram.array.copy()
        if arr is None: arr = self.X.copy()
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
        self.num_variables = num_variables
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
        # # Debug
        # print("[Debug:467] Metric tensors shapes?")
        # for mn, m in self.metrics_tensors().items():
        #     print(mn, m.shape)
        # # end Debug

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
                print(type(other_attr))
                if type(other_attr)!=type(attr_value): 
                    print("different types", attr_name)
                    return False
                if isinstance(attr_value, np.ndarray): 
                    if not (attr_value==other_attr).all(): 
                        print("different arrays", attr_name)
                        return False
                else:
                    if attr_value!=other_attr: 
                        print("different", type(attr_value), attr_name)
                        return False
        return True
    
    def empty_copy(self, k=1):
        # retrieve constructor args
        num_variables, num_groups, num_bins = self.histogram.shape
        assert num_bins % k == 0, f"resampling factor must give an integer, i.e. `num_bins/k` must be an integer"
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
        return rcu
    
    def upsample(self, k: int):
        """
        1. Compute bin mapping
        2. upsample histogram -> sum joined bins
        3. upsample metrics_tensors -> agg joined bins
        4. (optional) upsample other useless tensors (e.g. mean_variance)
        """
        # create empty copy
        rcu = self.empty_copy(k=k)
        # compute
        num_variables, num_groups, num_bins = rcu.histogram.shape
        bin_map = [np.arange(i, i+k) for i in np.arange(0, num_bins, k)]
        # upsample histogram
        for i, bins in enumerate(bin_map):
            rcu.histogram[:,:,i] = np.nansum(self.histogram[:,:,bins], axis=2)
        # upsample metrics
        for i, bins in enumerate(bin_map):
            for metric, metric_tensor in self.metrics_tensors():
                X = rcu.__getattribute__(metric)
                args = (self.histogram[:,:,bins],)
                if DualStratifiedMetric in metric.__class__.mro(): args += (metric_tensor.X1[:,:,bins], metric_tensor.X2[:,:,bins])
                else: args += (metric_tensor.X[:,:,bins],)
                X[:,:,i] = metric_tensor.agg(*args)
                rcu.__setattr__(metric, X)
        # upsample mean_variance
        for i, bins in enumerate(bin_map):
            rcu.mean_variance[:,:,i] = weighted_avg(self.mean_variance[:,:,bins], self.histogram[:,:,bins], axis=2)
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
        # compute mean diffs for mean error metrics
        error_kwargs = {}
        for metric_name, metric in [("mse", self.mse), ("rmse", self.rmse), ("mae", self.mae), ("mbe", self.mbe)]:
            values = reduce_binned(binned_diff, lambda x: np.nanmean(metric.fn(x)))
            error_kwargs[metric_name] = values
            metric.add(index, values)
        # add metrics
        for metric in self.metrics_tensors().values():
            # get metric kwargs
            kwargs = {
                k:v for k,v in list(self.kwargs().items())+[("binned_variance", binned_variance),("binned_diff", binned_diff),("counts", self.histogram[:,index,:])]
                if k in filter(lambda x: x!="self", inspect.getfullargspec(metric.evaluate_binned).args)
            }
            # add metric
            values = metric.evaluate_binned(**kwargs)
            if DualStratifiedMetric in metric.__class__.mro(): 
                metric.add(index, values[0], values[1])
            else: 
                metric.add(index, values)
        self.counter += 1
    
    def get(self, metric_names: Optional[List[str]]=None):
        """eg rcu.get(["mse", "mbe"])"""
        # get metrics
        if metric_names is None: metric_names = list(self.metrics_tensors().keys())
        metrics = [self.metrics_tensors()[mid] for mid in metric_names]
        # compute
        results = {}
        for metric, metric_name in zip(metrics, metric_names):
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
            print("[debug:546] Getting:", metric)
            res =  metric.get_subset(self.histogram, indexes)
            for k, v in res.items():
                res[k] = v.reshape(self.num_variables)
            results[metric_name] = res
            print(f"[debug:624] values: {results[metric_name]['values'].shape}, ause: {results[metric_name]['ause'].shape}")
        return results

class StratifiedRCUSubset(StratifiedRCU):
    def __init__(self, metric_names, *args, **kwargs):
        self.metric_names = metric_names
        super().__init__(*args, **kwargs)
    def metrics_tensors(self):
        return {k:v for k,v in super().metrics_tensors().items() if k in self.metric_names}
