import numpy as np
from itertools import chain

REGRESSION_METRICS = ["mae", "mae_p", "rmse", "rmse_p", "mbe", "mbe_p", "nll"]
UQ_METRICS = ["uce", "uce_p", "ence", "ence_p"]#, "auce", "c90", "r_v", "c_v", "sharpness"]
VARIABLE_NAMES = ['P95', 'MeanH', 'Dens', 'Gini', 'Cover']
EAST = ['346', '9', '341', '354', '415', '418', '416', '429', '439', '560', '472', '521', '498',
        '522', '564', '764', '781', '825', '796', '805', '827', '891', '835', '920', '959', '1023', '998',
        '527', '477', '542', '471']
WEST = ['528', '537', '792', '988', '769']
NORTH = ['819', '909', '896']
GROUPS = {"east": EAST, "west": WEST, "north": NORTH}
GLOBAL = EAST+WEST+NORTH

# Regression metrics
def mae(diff, *args, **kwargs): return np.abs(diff).mean(1)
def mse(diff, *args, **kwargs): return (diff**2).mean(1)
def mbe(diff, *args, **kwargs): return diff.mean(1)

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

class RUQMetrics:
    def __init__(
        self, 
        n_bins,
        regression_metrics=REGRESSION_METRICS,
        uq_metrics=UQ_METRICS,
        groups=GROUPS,
        variable_names=VARIABLE_NAMES
    ):
        # attributes
        self.n_bins = n_bins
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
        mean_mses, mean_vars, histogram = self.binned_variance(diff, variance)
        self.metrics[project_id] = {
            m: eval(m)(diff=diff, variance=variance, n_bins=self.n_bins)
            for m in chain(self.uq_metrics, self.regression_metrics)
        }
        # update count
        self.counts[project_id] = int(mask.sum())
        self.histograms[project_id] = histogram.astype(np.uint8)
        
    def aggregate_metrics(self):
        # compute CI for project statistics
        raise NotImplementedError("project CI must be implemented")
        # fill groups if needed
        for group_id in self.groups:
            raise NotImplementedError("group aggregation mut be implemented")
        # fill global if needed
        raise NotImplementedError("global aggregation mut be implemented")
    
    def binned_variance(self, diff, variance):
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
        n_bins = self.n_bins
        d = diff.shape[0]
        histogram = np.empty((d, n_bins))
        mean_vars = np.empty((d, n_bins))
        mean_mses = np.empty((d, n_bins))
        # loop on variables
        for i, (var, eps) in enumerate(zip(variance, diff)):
            # Bin variance linearly and get sample indexes
            bins = np.linspace(var.min(), var.max(), n_bins)
            bins_ids = np.digitize(var, bins=bins)
            # loop on bins to compute stats
            for bin_id in np.unique(bins_ids):
                # bin mask
                mask = bins_ids==bin_id
                bin_var, bin_diff = var[mask], eps[mask]
                # stats
                histogram[i,bin_id-1] = mask.astype(np.float16).sum()
                mean_vars[i,bin_id-1] = bin_var.mean()
                mean_mses[i,bin_id-1] = (bin_diff**2).mean()
        return mean_mses, mean_vars, histogram