import numpy as np
import torch
from torch import Tensor
from sklearn.linear_model import LinearRegression
from scipy import stats
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from typing import *
from osgeo import gdal
sns.set()
sns.set_style("whitegrid")

class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor


def all_same(array: np.ndarray):
    """
    Checks whether all elements in the given array have the same value
    """
    return (array.flat[0] == array.flat).all()


def nanmean(v, nan_mask, inplace=True, **kwargs):
    """
    https://github.com/pytorch/pytorch/issues/21987#issuecomment-539402619
    """
    if not inplace:
        v = v.clone()
    v[nan_mask] = 0
    return v.sum(**kwargs) / (~nan_mask).float().sum(**kwargs)


def split_list(_list, sep):
    result, sub = [], []
    for x in _list:
        if x == sep:
            if sub:
                result.append(sub)
                sub = []
        else:
            sub.append(x)
    if sub:
        result.append(sub)
    return result


def limit(tensor: Tensor, max=10) -> Tensor:
    """
    Clamp tensor below specified limit. Useful for preventing unstable training when using logarithmic network outputs.
    """
    return torch.clamp(tensor, max=max)


class RunningStats:
    """Efficiently keeps track of mean and standard deviation of a set of observations"""

    def __init__(self, shape: tuple):
        if not isinstance(shape, tuple):
            raise ValueError('shape must be tuple')
        self.shape = shape
        self.num_seen = 0
        self.mean = np.zeros(shape, dtype=np.float64)
        self.mean_of_squared = np.zeros(shape, dtype=np.float64)

    def add(self, data):
        assert data.shape[1:] == self.shape
        # cast to float64 to prevent overflows in next lines
        data = data.astype(np.float64)
        self.mean = (self.num_seen * self.mean + data.sum(0)) / (self.num_seen + data.shape[0])
        self.mean_of_squared = (self.num_seen * self.mean_of_squared + (data ** 2).sum(0)) / (self.num_seen + data.shape[0])
        self.num_seen += data.shape[0]

    @property
    def variance(self):
        return self.mean_of_squared - self.mean**2

    @property
    def std(self):
        return np.sqrt(self.variance)

    def to_dict(self):
        return {
            "num_seen": self.num_seen,
            "mean": self.mean.tolist(),
            "mean_of_squared": self.mean_of_squared.tolist(),
            "variance": self.variance.tolist(),
            "std": self.std.tolist()
        }
    
class CombinedStats(RunningStats):
    """Combine multiple RunningStats"""
    def __init__(self, shape):
        super().__init__(shape)
    
    def add(self, data):
        assert isinstance(data, dict)
        # float64 arrays
        new_num_seen = data["num_seen"]
        new_mean = np.array(data["mean"], dtype=np.float64)
        new_mean_of_squared = np.array(data["mean_of_squared"], dtype=np.float64)
        # update
        self.mean = (self.num_seen*self.mean+new_num_seen*new_mean)/(self.num_seen+new_num_seen)
        self.mean_of_squared = (self.num_seen*self.mean_of_squared+new_num_seen*new_mean_of_squared)/(self.num_seen+new_num_seen)
        self.num_seen += new_num_seen

class SpatialCorrelationAnalyzer:
    def __init__(
        self,
        save_dir=None
    ):
        self.save_dir = save_dir

    def fit(self, gt, pred):
        assert pred.shape==gt.shape
        self.pred = pred
        self.gt = gt
        self.res = gt-pred
        self.num_variables = pred.shape[0]
        self.gt_decorr, self.pred_decorr = self.decorrelate(gt.copy(), pred.copy())
        self.res_decorr = self.gt_decorr - self.pred_decorr
    
    def decorrelate(self, gt, pred):
        """
        Fits linear regression and rescale by value of linear prediction to spatially decorrellate
        """
        mask = ~np.isnan(pred).all(0)
        locations = np.indices(pred.shape[1:], dtype=float)
        locations[0,:,:] /= float(locations.shape[1]-1)
        locations[1,:,:] /= float(locations.shape[2]-1)
        x = locations[:,mask].reshape(-1,2)
        y = pred[:,mask].reshape(-1,self.num_variables)
        model = LinearRegression().fit(x,y)
        scaling = model.predict(locations.reshape(-1,2)).reshape(pred.shape)
        print(scaling)
        return gt-scaling, pred-scaling

    def plot_residuals(self, kind, variable_names):
        assert kind in ["original", "decorrelated"]
        if kind=="original":
            res_ = self.res
            pred_ = self.pred
        else:
            res_ = self.res_decorr
            pred_ = self.pred_decorr
        for k in range(self.num_variables):
            res = res_[:,~np.isnan(pred_).all(0)][k]
            pred = pred_[:,~np.isnan(pred_).all(0)][k]
            print(res.shape, pred.shape)
            gauss_fit_dist = stats.norm(loc=np.nanmean(res), scale=np.nanstd(res))
            gauss_sample = gauss_fit_dist.rvs(size=len(res))
            fig, axs = plt.subplots(nrows=3, ncols=1, figsize=(9,4))
            axs.flatten()
            # pred vs res
            axs[0].scatter(pred, res, alpha=.3, s=0.7)
            axs[0].set(xlabel="predictions", ylabel="residuals")
            # res histogram
            sns.histplot(gauss_sample, label="gaussian fit", ax=axs[1])
            sns.histplot(res, label="residuals", ax=axs[1])
            axs[1].legend(loc="upper left")
            # qq
            stats.probplot(res, dist=gauss_fit_dist, plot=axs[2], rvalue=True)
            axs[2].set_title("")
            # ks
            pval = stats.kstest(res, gauss_fit_dist.cdf).pvalue
            fig.suptitle(variable_names[k]+f"(p={pval:.3e})")
            plt.tight_layout()
            plt.show()

def compute_rasters_offsets(ds1, ds2):
    """
    Compute index offset between two GTiff rasters with the same CRS

    args
        ds1: reference GDal dataset
        ds2: other GDal dataset

    returns
        di: row offset, i.e. row of origin of ds2 in ds1 coordinate system
        dj: column offset, i.e. column of origin of ds2 in ds1 coordinate system
    """
    assert ds1.GetProjectionRef()==ds2.GetProjectionRef()
    # Get transforms
    Ox1, pw1, b1, Oy1, d1, ph1 = ds1.GetGeoTransform()
    Ox2, pw2, b2, Oy2, d2, ph2 = ds2.GetGeoTransform()
    assert (pw1==pw2 and ph1==ph2)
    # get offsets
    di = Ox2-Ox1
    dj = Oy2-Oy1
    return di, dj

def offset_rasters_valid_pixel(i, j, di, dj, other_h, other_w):
    """
    Checks wether the pixel at position (i,j) in a given raster
    falls inside an other raster

    args
        i, j: pixel coordinate in the reference raster system
        di, dj: position of origin of other raster in reference
            raster system
        other_h, other_w: size of other raster
    
    returns
        bool: true if (i,j) is in the other raster bounds
    """
    return (di<=i<=other_h+di and dj<=j<=other_w+dj)

def is_valid_rasters_offsets(i, j, gt_ds, s2_ds):
    di, dj = compute_rasters_offsets(gt_ds, s2_ds)
    s2_h, s2_w = s2_ds.RasterXSize, s2_ds.RasterYSize
    return offset_rasters_valid_pixel(i, j, di, dj, s2_h, s2_w)
