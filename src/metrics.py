import numpy as np
from itertools import chain
import scipy.stats
import yaml
from copy import deepcopy

REGRESSION_METRICS = ["mse", "mae", "rmse", "mbe", "nll"]
UQ_METRICS = ["uce", "ence", "auce", "r09", "c_v", "srp", "ause_rmse", "ause_uce"]
METRICS = list(chain(REGRESSION_METRICS, UQ_METRICS))
UQ_METRICS = ["uce", "ence", "auce", "r09", "c_v", "srp", "ause_rmse", "ause_uce"]

VARIABLE_NAMES = ['P95', 'MeanH', 'Dens', 'Gini', 'Cover']
EAST = ['346', '9', '341', '354', '415', '418', '416', '429', '439', '560', '472', '521', '498',
        '522', '564', '764', '781', '825', '796', '805', '827', '891', '835', '920', '959', '1023', '998',
        '527', '477', '542', '471']
WEST = ['528', '537', '792', '988', '769']
NORTH = ['819', '909', '896']
GLOBAL = ["west", "east", "north"]
GROUPS = {"east": EAST, "west": WEST, "north": NORTH, "global": GLOBAL}


def weighted_avg(values, counts, axis):
    return np.nansum(values*counts, axis=axis)/np.nansum(counts, axis=axis)

# Binned variance
def binned_variance(diff, variance, n_bins, var_mins=None, var_maxs=None, *args, **kwargs):
    """
    Compute mean MSE and mean variance in linear bins

    Args:
    - variance (np.ndarray[d, n]): predicted variance for each variable (d) and each pixel in the dataset (n)
    - diff (np.ndarray[d, n]): signed errors for each variable (d) and each pixel in the dataset (n)
    - n_bins (int): number of bins

    Returns:
    - mean_mses (np.ndarray[n_bins]): mean MSE in each bin
    - mean_vars (np.ndarray[n_bins]): mean variance in each bin
    - histogram (np.ndarray[n_bins]): number of samples in each bin
    """
    # initialize
    d = diff.shape[0]
    histogram = np.empty((d, n_bins))
    mean_vars = np.empty((d, n_bins))
    mean_mses = np.empty((d, n_bins))
    # loop on variables
    for i, (var, eps) in enumerate(zip(variance, diff)):
        # Bin variance linearly and get sample indexes
        var_min = var_mins[i] if var_mins is not None else var.min()
        var_max = var_maxs[i] if var_maxs is not None else var.max()
        bins = np.linspace(var_min, var_max, n_bins)
        bins_ids = np.digitize(var, bins=bins)-1 # digitize is in [1, n_bins] => -1 makes [0, n_bins-1]
        # loop on bins to compute stats
        for bin_id in range(n_bins):
            # bin mask
            mask = bins_ids==bin_id
            bin_var, bin_diff = var[mask], eps[mask]
            # stats
            histogram[i,bin_id] = mask.sum().astype(np.float32)
            if histogram[i,bin_id]!=0: 
                mean_vars[i,bin_id] = bin_var.mean()
                mean_mses[i,bin_id] = (bin_diff**2).mean()
            else: # empty bin!! set to NaN
                mean_vars[i,bin_id] = np.nan
                mean_mses[i,bin_id] = np.nan
    # make sure the only nan and zeros correspond to each others
    assert (np.isnan(mean_mses)==np.isnan(mean_vars)).all() and (np.isnan(mean_vars)==(histogram==0.)).all()
    return mean_mses, mean_vars, histogram

def area_under_spline(x, y):
    return 0.5*((x[:,1:]-x[:,:-1])*(y[:,:-1]+y[:,1:])).sum(1)

def sparsification_curve_x(variance, x, m, arr_dict):
    """arr_dict contain arrays on which to compute x"""
    N = variance.shape[1]
    nus = np.array([k*m for k in range(int(1/m)+1)])
    # sparsification curve
    sc = []
    # sort indexes by increasing variance
    sorted_idx = np.argsort(variance, axis=1)
    # sort argument arrays
    for k, v in arr_dict.items():
        arr_dict[k] = np.take_along_axis(v, sorted_idx, axis=1)
    for nu in nus:
        idxs = np.take_along_axis(sorted_idx, sorted_idx)
        # create sub_arrays
        sub_arr_dict = {}
        for k, v in arr_dict.items():
            sub_arr_dict[k] = v[:,:int(nu*N)]
        sc.append(x(**sub_arr_dict))
    # make nus, sc array for AU
    sc = np.stack(sc, axis=1) # (d, 1/m)
    nus = np.stack([nus for _ in range(5)], axis=0) # (d, 1/m)
    return nus, sc

# Regression metrics
def mae(diff, *args, **kwargs): return np.abs(diff).mean(1)
def mse(diff, *args, **kwargs): return (diff**2).mean(1)
def rmse(diff, *args, **kwargs): return np.sqrt((diff**2).mean(1))
def mbe(diff, *args, **kwargs): return diff.mean(1)

def nll(diff, variance, nll_eps, *args, **kwargs):
    v = variance
    v[v<nll_eps] = nll_eps # avoid numerical issues
    return 0.5 * (np.log(v) + (diff**2)/v).mean(1)

# UQ metrics
def uce(mean_mses, mean_vars, histogram, *args, **kwargs): 
    N = histogram[0].sum()
    return np.nansum(histogram*np.abs(mean_vars-mean_mses), axis=1)/N

# def ence(mean_rmses, mean_stds, histogram, *args, **kwargs):
    # N = histogram[0].sum()
    # return np.nansum(np.abs(mean_stds-mean_rmses)/mean_stds, axis=1)/N
def ence(mean_rmses, mean_stds, *args, **kwargs):
    return np.nansum(np.abs(mean_stds-mean_rmses)/mean_rmses, axis=1)

def empirical_accuracy(diff, variance, rho, *args, **kwargs):
    N = diff.shape[1]
    # intervals
    half_width = scipy.stats.norm.ppf((1+rho)/2)*np.sqrt(variance) # (d,n)
    # compute acc
    empirical_acc = np.count_nonzero(np.abs(diff)<half_width, axis=1)/N
    return empirical_acc

def auce(diff, variance, n_rho, expected_accs, return_acc=False, *args, **kwargs):
    d, N = diff.shape
    # accuracies
    empirical_accs = []
    for rho in expected_accs:
        # compute empirical acc
        empirical_acc = empirical_accuracy(diff, variance, rho)
        empirical_accs.append(np.expand_dims(empirical_acc, axis=1))
    # make arr
    empirical_accs = np.concatenate(empirical_accs, axis=1) # (d,n_rho)
    expected_accs = np.stack([expected_accs]*d, axis=0)
    # compute AU
    au = area_under_spline(x=expected_accs, y=np.abs(expected_accs-empirical_accs))
    if return_acc: return au, empirical_accs
    else: return au

def r09(diff, variance, *args, **kwargs): 
    return empirical_accuracy(diff, variance, rho=0.9)

def c_v(std, mean_std, *args, **kwargs):
    n = std.shape[1]
    mv = np.expand_dims(mean_std, axis=1) # (d,1)
    return np.sqrt(1/(n-1)*((std-mv)**2).sum(1))/mean_std
    # return np.sqrt(((std-mv)**2).sum(1)/(n-1))/mv.squeeze(1)

def srp(variance, *args, **kwargs):
    return variance.mean(1)

def ause(nus, sc):
    return area_under_spline(nus, sc)

def ause(variance, metric, ause_m, *args, **kwargs):
    """
    Returns metric in groups of decreasing maximum variance
    """
    # remove binned kwargs
    [kwargs.pop(kw, None) for kw in ["mean_mses", "mean_vars", "histogram"]]
    # dims
    d, n = variance.shape
    n_rm = int(n*ause_m)
    num_groups = int(n/n_rm)
    # sparsification curve
    sc = [] 
    # sort all arrays by variance index
    sorted_idx = np.argsort(variance, axis=1)
    variance = np.take_along_axis(variance, sorted_idx, axis=1)
    for arg in args:
        if isinstance(arg, np.ndarray): 
            arg = np.take_along_axis(arg, sorted_idx, axis=1)
    for k, v in kwargs.items():
        if isinstance(v, np.ndarray) and k!="labels_mean":
            kwargs[k] = np.take_along_axis(v, sorted_idx, axis=1)
    # compute ause in each group (from all variance to little variance)
    for i in range(num_groups):
        # group indexes
        group_idx = np.arange(0, n-i*n_rm)
        # select groups
        tmp_var = variance[:,group_idx]
        tmp_args = [
            a[:,group_idx] if isinstance(a, np.ndarray) else a
            for a in args
        ]
        tmp_kwargs = {
            k: v[:,group_idx] if isinstance(v, np.ndarray) and k!="labels_mean" else v
            for k,v in kwargs.items()
        }
        tmp_kwargs["variance"]=tmp_var
        # get variance bin if needed
        if metric in  [uce, ence]:
            mean_mses, mean_vars, histogram = binned_variance(*tmp_args, **tmp_kwargs)
            tmp_kwargs["mean_vars"]=mean_vars
            tmp_kwargs["mean_mses"]=mean_mses
            tmp_kwargs["histogram"]=histogram
        err = metric(*tmp_args, **tmp_kwargs)
        if not len(err.shape)==2: err = np.expand_dims(err, axis=1)
        sc.append(err)
    # prepare curve
    sc.append(np.zeros((d,1)))
    y = np.concatenate(sc[::-1], axis=1) # (d, num_groups)
    x = np.stack([np.arange(0,1+ause_m,ause_m)]*d, axis=0)
    return area_under_spline(x,y)

def ause_rmse(variance, ause_m, *args, **kwargs):
    return ause(variance, rmse, ause_m, *args, **kwargs)

def ause_uce(variance, ause_m, *args, **kwargs):
    return ause(variance, uce, ause_m, *args, **kwargs)

class RUQMetrics:
    def __init__(
        self, 
        n_bins,
        labels_mean,
        nll_eps=1e-06,
        n_rho=100,
        rho_min=1e-3,
        rho_max=1e-3,
        ause_m=.05,
        regression_metrics=REGRESSION_METRICS,
        uq_metrics=UQ_METRICS,
        groups=GROUPS,
        variable_names=VARIABLE_NAMES
    ):
        # attributes
        self.n_bins = n_bins
        self.labels_mean = labels_mean
        self.nll_eps = nll_eps
        self.n_rho = n_rho
        self.rho_min = rho_min 
        self.rho_max = rho_max 
        self.ause_m = ause_m
        self.regression_metrics = regression_metrics
        self.uq_metrics = uq_metrics
        self.groups = groups
        self.variable_names = variable_names
        # accumulators: allow for running computations
        self.diffs = {}
        self.variances = {}
        self.metrics = {}
        self.groups = {group_id: [] for group_id in GROUPS.keys()}

    def add_project(self, project_id, mean, variance, gt):
        assert mean.shape == variance.shape == gt.shape
        assert mean.shape[0]==len(self.variable_names), f"arrays must be of shape (num_variables, num_samples)"
        # masked difference
        mask = ~np.isnan(mean).all(0)
        diff = mean[:,mask]-gt[:,mask]
        variance = variance[:,mask]
        self.diffs[project_id] = diff 
        self.variances[project_id] = variance
        # add group
        for group_id, group in GROUPS.items(): 
            if project_id in group: 
                self.groups[group_id].append(project_id)

    def fill_metrics(self, key, diff, variance):
        mean_mses, mean_vars, histogram = binned_variance(diff, variance, self.n_bins)
        all_kwargs = dict(diff=diff, variance=variance, n_bins=self.n_bins,
                    mean_mses=mean_mses, mean_vars=mean_vars, 
                    histogram=histogram, n_rho=self.n_rho,
                    ause_m=self.ause_m, nll_eps=self.nll_eps,
                    labels_mean=self.labels_mean, rho_min=self.rho_min,
                    rho_max=self.rho_max)
        self.metrics[key] = {
            m: eval(m)(**all_kwargs)
            for m in chain(self.uq_metrics, self.regression_metrics)
        }
        
    def aggregate_entity(self, project_id=None, group_id=None, compute_intermediary=True):
        assert not (project_id and group_id)
        # single project
        if project_id: entities, key = [project_id], project_id
        # group
        elif group_id: entities, key = self.groups[group_id], group_id
        # global aggregation
        else: entities, key = list(chain(*list(self.groups.values()))), "global"
        print("Aggregating entity: {} -> {} ()".format(key, entities, self.metrics.keys()))
        # compute only if not existing
        if not key in self.metrics.keys() and len(entities)>0:
            diff, variance = None, None
            for pid in entities:
                diff_, variance_ = self.diffs[pid] , self.variances[pid]
                diff = diff_ if diff is None else np.concatenate([diff, diff_], axis=1)
                variance = variance_ if variance is None else np.concatenate([variance, variance_], axis=1)
                # compute intermediary if needed
                if compute_intermediary and not pid in self.metrics.keys():
                    print("Computing single entity: {}".format(pid))
                    self.fill_metrics(pid, diff_.copy(), variance_.copy())
            # compute entity metrics if not yet done
            if len(entities)>1: self.fill_metrics(key, diff, variance)

    def aggregate_all(self):
        _ = self.aggregate_entity(compute_intermediary=True)
        for group_id in GROUPS.keys(): _ = self.aggregate_entity(group_id=group_id, compute_intermediary=False)

class OnlineRUQMetrics:
    """
    Online metrics:
        - Error-based (mse, mae, rmse, mbe): E = (f(diff) + N*E)/(N+1); N+=1 (w/ diff=(gt-mu)**2)
        - NLL                              : L = (l + 2*N*L)/(2*(N+1)); N+=1 (w/ l=log(var)+diff**2/var)
        - binned variance-based (uce, ence): 
            - Precompute a high resolution histogram: H_hi = {|B_hi(k)|: k=0,...,M_hi-1} between pred_var_max and pred_var_min (across all predicitions)
            - When computing a metric, 
                - Downsample to get *_lo(i), i=0,...,M_lo-1
                    - Constant bin width: m = M_hi/M_lo
                        - |B_lo(i)| = sum_{k=0,...,m-1}|B_hi(k+m*i)|
                        - mean_x_lo(i) = 1/|B_lo(i)|*|sum_{k=0,...,m-1}|B_hi(k+m*i)|*mean_x_hi(i+m*k)
                    - Almost constant bin density:
                        - TODO
                - Compute metric on downsampled arrays
        - ause                             :
            - Use the high resolution histogram. ause_m is used to select a number of bins to remove.
            - Remove the required bins and compute the metric by unbinning (i.e. downsampling to n_bins=1) the remaining bins
    """
    def __init__(
        self,
        # High resolution binning
        var_lo,
        var_hi,
        n_bins,
        # NLL
        nll_eps=1e-6,
        # AUCE
        n_rho=100,
        rho_min=1e-3,
        rho_max=1-1e-3,
        # AUSE
        ause_step=.05,
    ):
        self.var_lo = var_lo
        self.var_hi = var_hi
        self.n_bins = n_bins
        self.nll_eps = nll_eps
        self.n_rho = n_rho
        self.rho_min = rho_min
        self.rho_max = rho_max 
        self.ause_step = ause_step
        # accumulators
        self.metrics = {}
        self.counts = {}
        self.bin_mses = {}
        self.bin_vars = {}
        self.histogram = {} # hi resolution histogram (lo resolution is computed on demand)
        # internal global variables
        self.error_metrics = ["mse", "mae", "rmse", "mbe"]
        self.methods_suffix = ["errors", "nll", "ence", "auce", "r09", "c_v", "srp", "ause_rmse", "ause_uce"]
        self.auce_rhos = np.linspace(rho_min, rho_max, n_rho)

    def add(self, project_id, means, gt, variance):
        assert means.shape == variance.shape == gt.shape
        assert means.shape[0]==len(self.variable_names), f"arrays must be of shape (num_variables, num_samples)"
        assert not project_id in self.metrics.keys(), f"{project_id} was already added"
        self.metrics[project_id] = {}
        # masked difference
        mask = ~np.isnan(means).all(0)
        diff = means[:,mask]-gt[:,mask] # (d,n)
        variance = variance[:,mask] # (d,n)
        # Counts
        self.counts[project_id] = diff.shape[1]
        # Binning
        p_mean_mses, p_mean_vars, p_histogram = binned_variance(diff, variance, self.n_bins, self.var_lo, self.var_hi)
        self.bin_mses[project_id] = p_mean_mses
        self.bin_vars[project_id] = p_mean_vars
        self.histogram[project_id] = p_histogram
        # Error based metrics
        self._add_errors(diff, self.metrics[project_id])
        # NLL
        self._add_nll(diff, variance, self.metrics[project_id])
        # uce
        self._add_uce(p_mean_mses, p_mean_vars, p_histogram, self.metrics[project_id])
        # ence
        p_mean_rmses = np.sqrt(p_mean_mses)
        p_mean_stds = np.sqrt(p_mean_vars)
        self._add_ence(p_mean_rmses, p_mean_stds, self.metrics[project_id])
        # auce
        self._add_auce(diff, variance, self.metrics[project_id])
        # r09
        self._add_r09(diff, variance, self.metrics[project_id])
        # c_v
        self._add_c_v(variance, self.metrics[project_id])
        # srp
        self._add_srp(variance, self.metrics[project_id])
        # ause_rmse
        # ause_uce

    def agg(self, groups=GROUPS):
        for group_id, group_keys in groups.items():
            if not group_id in self.metrics.keys(): self.metrics[group_id] = {}
            # Add count
            self.counts[group_id] = np.stack([self.counts[k] for k in group_keys], axis=-1).sum(1)
            # Apply aggregation
            for method_suffix in self.methods_suffix:
                eval(f"self._agg_{method_suffix}")(group_id, group_keys)

    def resample_histogram(self, M_new, kind="linear"):
        if kind == "linear":
            raise NotImplementedError()
        else:
            raise ValueError(f"Invalid: kind={kind}")

    # Errors
    def _add_errors(self, diff, m_dict):
        d = diff.shape[0]
        for metric in self.error_metrics:
            m_dict[metric] = eval(metric)(diff)
            assert m_dict[metric].shape == (d,)

    def _agg_errors(self, group_id, group_keys):
        counts = np.stack([self.counts[k] for k in group_keys], axis=-1) # (d, P)
        for metric in self.error_metrics:
            errors = np.stack([self.metrics[k][metric] for k in group_keys], axis=-1) # (d, P) w. P: group size
            self.metrics[group_id][metric] = weighted_avg(errors, counts, axis=1) # (d,)

    # NLL
    def _add_nll(self, diff, variance, m_dict):
        d = diff.shape[0]
        m_dict["nll"] = nll(diff, variance, self.nll_eps)
        assert m_dict["nll"].shape == (d,)

    def _agg_nll(self, group_id, group_keys):
        losses = np.stack([self.metrics[k]["nll"] for k in group_keys], axis=-1) # (d, P) w. P: group size
        counts = np.stack([self.counts[k] for k in group_keys], axis=-1) # (d, P)
        self.metrics[group_id]["nll"] = weighted_avg(losses, counts, axis=1) # (d,)

    # uce
    def _add_uce(self, p_mean_mses, p_mean_vars, p_histogram, m_dict):
        d, M = p_mean_mses.shape
        assert M==self.n_bins 
        m_dict["uce"] = uce(p_mean_mses, p_mean_vars, p_histogram)
        assert m_dict["uce"].shape == (d,)

    def _agg_uce(self, group_id, group_keys):
        H = self.stack([self.histogram[k] for k in group_keys], axis=-1) # (d, M, P) w. M: n_bins, P: group size
        bin_mses = self.stack([self.bin_mses[k] for k in group_keys], axis=-1) # (d, M, P)
        bin_vars = self.stack([self.bin_vars[k] for k in group_keys], axis=-1) # (d, M, P)
        counts = np.stack([self.counts[k] for k in group_keys], axis=-1) # (d, P)
        assert counts.shape[0]==bin_mses.shape[0]==bin_vars.shape[0]==H.shape[0]
        d = counts.shape[0]
        assert (np.nansum(H,axis=2)==counts).all(), "unmatching histogram and counts"
        assert (np.nansum(H,axis=(1,2))==np.nansum(counts,axis=1)).all(), "unmatching histogram and counts"
        N = np.nansum(counts, axis=1)
        value = bin_mses-bin_vars
        value = H*value
        value = np.nansum(value, axis=2)
        value = np.abs(value)
        value = np.nansum(value, axis=1) / N
        assert value.shape == (d,)
        self.metrics[group_id]["uce"] = value

    # ence
    def _add_ence(self, p_mean_rmses, p_mean_stds, m_dict):
        d, M = p_mean_rmses.shape
        assert M==self.n_bins
        m_dict["ence"] = ence(p_mean_rmses, p_mean_stds)
        assert m_dict["ence"].shape == (d,)

    def _agg_ence(self, group_id, group_keys):
        H = self.stack([self.histogram[k] for k in group_keys], axis=-1) # (d, M, P) w. M: n_bins, P: group size
        bin_mses = self.stack([self.bin_mses[k] for k in group_keys], axis=-1) # (d, M, P)
        bin_vars = self.stack([self.bin_vars[k] for k in group_keys], axis=-1) # (d, M, P)
        counts = np.stack([self.counts[k] for k in group_keys], axis=-1) # (d, P)
        assert counts.shape[0]==bin_mses.shape[0]==bin_vars.shape[0]==H.shape[0]
        d = counts.shape[0]
        assert (np.nansum(H,axis=2)==counts).all(), "unmatching histogram and counts"
        assert (np.nansum(H,axis=(1,2))==np.nansum(counts,axis=1)).all(), "unmatching histogram and counts"
        N = np.nansum(counts, axis=1)
        value = np.sqrt(np.nansum(H*bin_vars, axis=2))
        value -= np.sqrt(np.nansum(H*bin_mses, axis=2))
        value /= np.sqrt(np.nansum(H*bin_vars, axis=2))
        value = np.nansum(value, axis=1)/N 
        assert value.shape == (d,)
        self.metrics[group_id]["ence"] = value

    # auce
    def _add_auce(self, diff, variance, m_dict):
        d = diff.shape[0]
        ause_rhos = np.linspace(self.rho_min, self.rho_max, self.n_rho) # (d,)
        auce_, empirical_accs = auce(diff, variance, ause_rhos, return_accs=True)
        assert auce_.shape == (d,)
        assert empirical_accs.shape[0] == d
        m_dict["empirical_accs"] = empirical_accs
        m_dict["auce"] = auce_

    def _agg_auce(self, group_id, group_keys):
        empirical_accs = np.stack([self.metrics[k]["empirical_accs"] for k in group_keys], axis=1) # (d, P, n_rho) w. P: group size
        counts = np.expand_dims(np.stack([self.counts[k] for k in group_keys], axis=-1), axis=-1) # (d, P, 1)
        empirical_accs = weighted_avg(empirical_accs, counts, axis=1) # (d, n_rho)
        d = counts.shape[0]
        expected_accs = np.stack([np.linspace(self.rho_min, self.rho_max, self.n_rho)]*d, axis=0)
        assert expected_accs.shape==empirical_accs.shape
        self.metrics[group_id]["auce"] = area_under_spline(x=expected_accs, y=np.abs(expected_accs-empirical_accs))
    
    # r09
    def _add_r09(self, diff, variance, m_dict):
        d = diff.shape[0]
        m_dict["r09"] = r09(diff, variance)
        assert m_dict["r09"].shape[0]==(d,)

    def agg_r09(self, group_id, group_keys):
        empirical_accs = np.stack([self.metrics[k]["empirical_accs"] for k in group_keys], axis=1) # (d, P) w. P: group size
        counts = np.expand_dims(np.stack([self.counts[k] for k in group_keys], axis=-1), axis=-1) # (d, P)
        self.metrics[group_id]["r09"] = weighted_avg(empirical_accs, counts)

    # ause_x
    def _add_ause_x(self, variance, x, arr_dict, m_dict):
        d = variance.shape[0]
        scurve = sparsification_curve_x(variance, x, arr_dict)
        m_dict[f"ause_{x.__name__}"] = ause(*scurve)
        assert m_dict[f"ause_{x.__name__}"].shape == (d,)
    def _agg_ause_x(self, variance, x, arr_dict): pass

    # c_v
    def _add_c_v(self, variance, m_dict):
        d = variance.shape[0]
        std = np.sqrt(variance) # (d,n)
        mean_std = std.mean(1) # (d,)
        m_dict["c_v_mean_std"] = mean_std
        m_dict["c_v"] = c_v(std, mean_std) # (d,)
        assert m_dict["c_v"].shape == (d,)

    def _agg_c_v(self, group_id, group_keys):
        m_stds = np.stack([self.metrics[k]["c_v_mean_std"] for k in group_keys], axis=-1) # (d, P) w. P: group size
        c_vs = np.stack([self.metrics[k]["c_v"] for k in group_keys], axis=-1) # (d, P)
        counts = np.stack([self.counts[k] for k in group_keys], axis=-1) # (d, P)
        m_std = weighted_avg(m_stds, counts, axis=1)
        self.metrics[group_id]["c_v_mean_std"] = m_std
        self.metrics[group_id]["c_v"] = np.sqrt(c_vs**2*m_stds**2*(counts-1)/((counts.sum(1)-1)))/m_std

    # srp
    def _add_srp(self, variance, m_dict):
        d = variance.shape[0]
        m_dict["srp"] = srp(variance)
        assert m_dict["srp"].shape == (d,)

    def _agg_srp(self, group_id, group_keys):
        srps = np.stack([self.metrics[k]["srp"] for k in group_keys], axis=-1) # (d, P) w. P: group size
        counts = np.stack([self.counts[k] for k in group_keys], axis=-1) # (d, P)
        self.metrics[group_id]["srp"] = weighted_avg(srps, counts, axis=1) # (d,)

def main(N_projects):
    from pathlib import Path
    import rasterio
    import json

    PREDICTIONS_DIR = Path("results/dev/2023-03-14_15-45-23")
    PKL_DIR = Path('data/pkl/2021-05-18_10-57-45')
    GT_DIR = Path('data/preprocessed')

    with (PKL_DIR / 'stats.yaml').open() as fh:
        # load training set statistics for data normalization
        stats = yaml.safe_load(fh)
        labels_mean = np.array(stats['labels_mean'])

    ruq_metrics = RUQMetrics(n_bins=20, labels_mean=labels_mean)

    i = 0
    for mean_file in PREDICTIONS_DIR.glob("*_mean.tif"):
        project = mean_file.stem.split("_")[0]
        if project not in GLOBAL:
            continue
        with rasterio.open(mean_file) as fh:
            mean = fh.read(fh.indexes) #/np.expand_dims(labels_mean, axis=(1,2))
        with rasterio.open(PREDICTIONS_DIR / (project + '_variance.tif')) as fh:
            variance = fh.read(fh.indexes) #/np.expand_dims(labels_mean, axis=(1,2))
        with rasterio.open(GT_DIR / (project + '.tif')) as fh:
            gt = fh.read(fh.indexes)
            gt_mask = fh.read_masks(1).astype(bool) 
        print(f"Adding {project}")
        ruq_metrics.add_project(project, mean, variance, gt)
        i += 1
        if i==N_projects: break
        
    ruq_metrics.aggregate_all()
    for entity, em in ruq_metrics.metrics.items():
        print(f"{entity}:")
        for m, vs in em.items():
            print(f"    {m}: {', '.join(['{:.3f}'.format(v) for v in vs])}")
 
def test(d=2, verbose=True):
    # dims
    d, N = (d,30)
    # params
    nll_eps = 1e-10
    n_bins = 2
    n_rho = 3
    rho_min = 1e-3
    rho_max = 1-1e-3
    rho = np.linspace(rho_min, rho_max, n_rho)
    nd_rho = np.stack([rho]*d, axis=0)
    ause_m = .5
    # generate data
    eps = 10*np.random.randn(d,1)
    gt = np.random.randn(d, N)
    mean = gt + eps
    variance = np.concatenate([np.exp(1)*np.ones((d, int(N/2))), np.exp(2)*np.ones((d, int(N/2)))], axis=1)
    assert (variance>nll_eps).all()
    var_min = np.exp(1)*np.ones((d,1))
    var_max = np.exp(2)*np.ones((d,1))
    std_min = np.sqrt(var_min)
    std_max = np.sqrt(var_max)
    std_mean = (std_min+std_max)/2
    labels_mean = 2*np.ones((d, 1))
    # binning
    true_binning = np.array([eps**2, eps**2]).squeeze(-1).transpose(), np.concatenate([var_min, var_max], axis=1), np.array([[N/2, N/2]]*d)
    pred_binning = binned_variance(mean-gt, variance, n_bins)
    if verbose:
        for name, true, pred in zip(["mean_mses", "mean_vars", "histogram"], true_binning, pred_binning):
            print("[{}: {}] target={}, computed={}".format(name, np.allclose(pred, true), list(true), list(pred)))
    # compute metrics
    ruq = RUQMetrics(n_bins=n_bins, labels_mean=labels_mean, nll_eps=nll_eps, 
                     rho_min=rho_min, rho_max=rho_max, n_rho=n_rho, ause_m=ause_m,
                     variable_names=["dumb"]*d)
    ruq.add_project(project_id=EAST[0], mean=mean, variance=variance, gt=gt)
    ruq.aggregate_entity(EAST[0])
    # gt metrics
    expected_accs = 0.5*(
        (np.abs(eps)<scipy.stats.norm.ppf((1+rho)/2)*np.sqrt(var_min)).astype(np.float32) + \
        (np.abs(eps)<scipy.stats.norm.ppf((1+rho)/2)*np.sqrt(var_max)).astype(np.float32)
    )
    gt_metrics = dict(
        mse = eps**2,
        mae = np.abs(eps),
        rmse = np.abs(eps),
        mbe = eps,
        nll = 1/4*(3+eps**2*(1/var_min+1/var_max)),
        uce = 0.5*(np.abs(var_min-eps**2)+np.abs(var_max-eps**2)),
        ence = (np.abs(var_min-eps**2)/var_min+np.abs(var_max-eps**2)/var_max)/N,
        auce = area_under_spline(nd_rho, np.abs(nd_rho-expected_accs)),
        r09 = 0.5*(
            (np.abs(eps)<scipy.stats.norm.ppf((1+.9)/2)*np.sqrt(var_min)).astype(np.float32) + \
            (np.abs(eps)<scipy.stats.norm.ppf((1+.9)/2)*np.sqrt(var_max)).astype(np.float32)
        ),
        c_v = np.sqrt(N*(std_max-std_min)**2/(4*(N-1)))/std_mean,
        srp = (var_min+var_max)/2,
        ause_rmse = area_under_spline(
            x=np.stack([np.arange(0,1+ause_m,ause_m)]*d, axis=0),
            y=np.stack([
                np.zeros((d,1)),
                np.abs(eps),
                np.abs(eps)
            ], axis=1).squeeze(-1)
        ),
        ause_uce = area_under_spline(
            x=np.stack([np.arange(0,1+ause_m,ause_m)]*d, axis=0),
            y=np.stack([
                np.zeros((d,1)),
                np.abs(var_min-eps**2),
                (np.abs(var_min-eps**2)+np.abs(var_max-eps**2))/2
            ], axis=1).squeeze(-1)
        )
    )
    # compare
    results = {}
    for k in gt_metrics.keys():
        gtm = gt_metrics[k] if len(gt_metrics[k].shape)==1 else gt_metrics[k].squeeze(1)
        rum = ruq.metrics[EAST[0]][k]
        if verbose: print("[{}: {}] target={}, computed={}, diff={}, ratio={}".format(k, np.allclose(gtm, rum), gtm, rum, list(gtm-rum), list(gtm/rum)))
        results[k] = {"same": np.allclose(gtm, rum), "error": list(gtm-rum)}
    return results

def multi_test():
    import json
    all_results = []
    num_runs = 10
    for _ in range(num_runs):
        for d in range(2, 10, 1):
            all_results.append(test(d, False))
    same_dict = {k: True for k in all_results[0].keys()}
    for results in all_results:
        for k in results.keys():
            same_dict[k] = results[k]["same"] and same_dict[k]
    print(json.dumps(same_dict, indent=2))

if __name__ == "__main__": 
    # main(2)
    multi_test()