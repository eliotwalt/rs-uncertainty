import numpy as np
from itertools import chain
import scipy.stats

REGRESSION_METRICS = ["mae", "mae_p", "rmse", "rmse_p", "mbe", "mbe_p", "nll"]
UQ_METRICS = ["uce", "uce_p", "ence", "ence_p", "auce", "r09", "c_v", "srp", "srp_p", "ause_rmse_p", "ause_uce_p"]
VARIABLE_NAMES = ['P95', 'MeanH', 'Dens', 'Gini', 'Cover']
EAST = ['346', '9', '341', '354', '415', '418', '416', '429', '439', '560', '472', '521', '498',
        '522', '564', '764', '781', '825', '796', '805', '827', '891', '835', '920', '959', '1023', '998',
        '527', '477', '542', '471']
WEST = ['528', '537', '792', '988', '769']
NORTH = ['819', '909', '896']
GROUPS = {"east": EAST, "west": WEST, "north": NORTH}
GLOBAL = EAST+WEST+NORTH

"""
TODO:
- test if setting mean_{mses, vars} to nan when histo == 0 creates issues (i.e. should use np.nansum, nanmean, etc. later?)
- test on known inputs!
- compute on all projects as binning cannot be undone :/
"""

# Binned variance
def binned_variance(diff, variance, n_bins, *args, **kwargs):
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
        histogram = -1*np.ones((d, n_bins))
        mean_vars = -1*np.ones((d, n_bins))
        mean_mses = -1*np.ones((d, n_bins))
        # loop on variables
        for i, (var, eps) in enumerate(zip(variance, diff)):
            # Bin variance linearly and get sample indexes
            bins = np.linspace(var.min(), var.max(), n_bins)
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
        assert (histogram>=0).all()
        return mean_mses, mean_vars, histogram

# Regression metrics
def mae(diff, *args, **kwargs): return np.abs(diff).mean(1)
def mse(diff, *args, **kwargs): return (diff**2).mean(1)
def rmse(diff, *args, **kwargs): return np.sqrt((diff**2).mean(1))
def mbe(diff, *args, **kwargs): return diff.mean(1)
def mse_p(diff, *args, **kwargs): return (diff**2).mean(1)/diff.max(1)
def rmse_p(diff, *args, **kwargs): return np.sqrt(rmse(diff, *args, **kwargs))
def mae_p(diff, *args, **kwargs): return mae(diff, *args, **kwargs)/diff.max(1)
def mbe_p(diff, *args, **kwargs): return mbe(diff, *args, **kwargs)/diff.max(1)

def nll(diff, variance, nll_eps, *args, **kwargs):
    # avoid numerical issues
    v = variance
    v[v<nll_eps] = nll_eps
    return 0.5 * (np.log(v) + (diff/v)**2).mean(1)

# UQ metrics
def uce(mean_mses, mean_vars, histogram, *args, **kwargs): 
    return (histogram*np.abs(mean_vars-mean_mses)).mean(1) 

def uce_p(mean_mses, mean_vars, histogram, *args, **kwargs): 
    return uce(mean_mses, mean_vars, histogram)/np.abs(mean_mses-mean_vars).max(1)

def ence(mean_mses, mean_vars, *args, **kwargs):
    return (np.abs(mean_vars-mean_mses)/mean_vars).mean(1)

def ence_p(mean_mses, mean_vars, histogram, *args, **kwargs):
    N = float(histogram[0].sum())
    M = float(histogram.shape[1])
    phis = (np.abs(mean_vars-mean_mses)/mean_vars).max(1)
    alphas = N*phis/M 
    return ence(mean_mses, mean_vars)*alphas

def empirical_accuracy(diff, variance, rho, *args, **kwargs):
    N = diff.shape[1]
    # intervals
    half_width = scipy.stats.norm.ppf((1+rho)/2)*variance # (d,n)
    # compute acc
    empirical_acc = np.count_nonzero(np.abs(diff)<half_width, axis=1)/N
    return empirical_acc

def auce(diff, variance, n_rho, *args, **kwargs):
    """
    From diff:
        -CDF^-1((rho+1)/2)*variance < diff < CDF^-1((rho+1)/2)*variance
        => |diff| < CDF^-1((rho+1)/2)*variance
    """
    N = diff.shape[1]
    # accuracies
    expected_accs = np.linspace(0, 1, n_rho) # (n_rho,)
    empirical_accs = []
    for rho in expected_accs:
        # compute empirical acc
        empirical_acc = empirical_accuracy(diff, variance, rho)
        empirical_accs.append(np.expand_dims(empirical_acc, axis=1))
    # make arr
    empirical_accs = np.concatenate(empirical_accs, axis=1) # (d,n_rho)
    # average across rho
    return np.abs(empirical_accs-expected_accs).mean(1) # (d,)

def r09(diff, variance, *args, **kwargs): 
    return empirical_accuracy(diff, variance, rho=0.9)

def c_v(variance, *args, **kwargs):
    n = variance.shape[1]
    mv = np.expand_dims(variance.mean(1), axis=1) # (d,1)
    # print("var var:", (variance-mv).sum(1))
    # print("n-1:", n-1)
    # print("no sqrt:", (variance-mv).sum(1)/(n-1))
    # print("sqrt:", np.sqrt((variance-mv).sum(1)/(n-1)))
    # print("mv.squeeze(1):", mv.squeeze(1))
    return np.sqrt(((variance-mv)**2).sum(1)/(n-1))/mv.squeeze(1)

def srp(variance, *args, **kwargs):
    return variance.mean(1)

def srp_p(variance, *args, **kwargs):
    return srp(variance, *args, **kwargs)/variance.max(1)

def ause(variance, metric, ause_m, *args, **kwargs):
    """
    Returns metric in groups of decreasing maximum variance
    """
    # remove binned kwargs
    [kwargs.pop(kw, None) for kw in ["mean_mses", "mean_vars", "histogram"]]
    # sparsification curve
    sc = [] 
    # dims
    d, n = variance.shape
    n_rm = int(n*ause_m)
    num_groups = int(n/n_rm)
    # sort all arrays by variance index
    sorted_idx = np.argsort(variance, axis=1)
    variance = np.take_along_axis(variance, sorted_idx, axis=1)
    for arg in args:
        if isinstance(arg, np.ndarray): 
            arg = np.take_along_axis(arg, sorted_idx, axis=1)
    for k, v in kwargs.items():
        if isinstance(v, np.ndarray):
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
        # tmp_kwargs = {}
        # for k,v in kwargs.items():
        #     if isinstance(v, np.ndarray):
        #         print("Reshaping: ", k, v.shape)
        #         tmp_kwargs[k] = v[:,group_idx]
        #     else:
        #         tmp_kwargs[k] = v
        tmp_kwargs = {
            k: v[:,group_idx] if isinstance(v, np.ndarray) else v
            for k,v in kwargs.items()
        }
        tmp_kwargs["variance"]=tmp_var
        # get variance bin if needed
        if metric in  [uce, uce_p, ence, ence_p]:
            mean_mses, mean_vars, histogram = binned_variance(*tmp_args, **tmp_kwargs)
            tmp_kwargs["mean_vars"]=mean_vars
            tmp_kwargs["mean_mses"]=mean_mses
            tmp_kwargs["histogram"]=histogram
        err = metric(*tmp_args, **tmp_kwargs)
        if not len(err.shape)==2: err = np.expand_dims(err, axis=1)
        sc.append(err)
    sc = np.concatenate(sc, axis=1) # (d, num_groups)
    return sc.mean(1) # riemann sum is just the mean

def ause_rmse_p(variance, ause_m, *args, **kwargs):
    return ause(variance, rmse_p, ause_m, *args, **kwargs)

def ause_uce_p(variance, ause_m, *args, **kwargs):
    return ause(variance, uce_p, ause_m, *args, **kwargs)

class RUQMetrics:
    def __init__(
        self, 
        n_bins,
        nll_eps=1e-06,
        n_rho=100,
        ause_m=.05,
        regression_metrics=REGRESSION_METRICS,
        uq_metrics=UQ_METRICS,
        groups=GROUPS,
        variable_names=VARIABLE_NAMES
    ):
        # attributes
        self.n_bins = n_bins
        self.nll_eps = nll_eps
        self.n_rho = n_rho
        self.ause_m = ause_m
        self.regression_metrics = regression_metrics
        self.uq_metrics = uq_metrics
        self.groups = groups
        self.variable_names = variable_names
        # accumulators: allow for running computations
        self.metrics = {}
        self.counts = {}
        self.histograms = {}

    def add_project(self, project_id, mean, variance, gt):
        assert mean.shape == variance.shape == gt.shape
        assert mean.shape[0]==len(self.variable_names), f"arrays must be of shape (num_variables, num_samples)"
        # masked difference
        mask = ~np.isnan(mean).all(0)
        diff = mean[:,mask]-gt[:,mask]
        variance = variance[:,mask]
        # compute metrics
        mean_mses, mean_vars, histogram = binned_variance(diff, variance, self.n_bins)
        self.metrics[project_id] = {
            m: eval(m)(diff=diff, variance=variance, n_bins=self.n_bins,
                       mean_mses=mean_mses, mean_vars=mean_vars, 
                       histogram=histogram, n_rho=self.n_rho,
                       ause_m=self.ause_m, nll_eps=self.nll_eps)
            for m in chain(self.uq_metrics, self.regression_metrics)
        }
        # update count
        self.counts[project_id] = int(mask.sum())
        self.histograms[project_id] = histogram.astype(np.int_)
        # inspect metrics
        
    def aggregate_metrics(self):
        # compute CI for project statistics
        raise NotImplementedError("project CI must be implemented")
        # fill groups if needed
        for group_id in self.groups:
            raise NotImplementedError("group aggregation mut be implemented")
        # fill global if needed
        raise NotImplementedError("global aggregation mut be implemented")

if __name__ == "__main__":
    from pathlib import Path
    import rasterio

    PREDICTIONS_DIR = Path("results/dev/2023-03-14_15-45-23")
    GT_DIR = Path('data/preprocessed')

    ruq_metrics = RUQMetrics(n_bins=20)
    
    projects = EAST[:2]

    for mean_file in PREDICTIONS_DIR.glob("*_mean.tif"):
        project = mean_file.stem.split("_")[0]
        if project not in projects:
            continue        
        with rasterio.open(mean_file) as fh:
            mean = fh.read(fh.indexes)
        with rasterio.open(PREDICTIONS_DIR / (project + '_variance.tif')) as fh:
            variance = fh.read(fh.indexes)
        with rasterio.open(GT_DIR / (project + '.tif')) as fh:
            gt = fh.read(fh.indexes)
            gt_mask = fh.read_masks(1).astype(bool) 
        
        ruq_metrics.add_project(project, mean, variance, gt)