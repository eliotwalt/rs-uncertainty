import wandb
import pandas as pd
import matplotlib.pyplot as plt 
import matplotlib.patches as patches
from matplotlib.ticker import MaxNLocator
from matplotlib.colors import LogNorm
import tempfile
import math
import json
from tqdm import tqdm
import os
import seaborn as sns
import rasterio
import fiona
from datetime import datetime
from pathlib import Path
import numpy as np
from .metrics import StratifiedRCU
sns.set()
sns.set_style("whitegrid")

# Compute gt date
def parse_gt_date(date_str) -> datetime: 
    # there are 2 gt date formats
    try: return datetime.strptime(date_str, "%Y-%m-%d")
    except: return datetime.strptime(date_str, "%Y/%m/%d")

def single_legend_figure(fig, ncol):
    objects, labels = fig.axes[0].get_legend_handles_labels()
    fig.legend(objects[:2], labels[:2], ncol=ncol, loc='upper center', bbox_to_anchor=(0.5, 0.))
    return fig

def compute_gt_date(project_id, shapefile_paths):
    project_shape_collections = [fiona.open(p) for p in shapefile_paths]
    polygon = None
    for collection in project_shape_collections:
        try:
            polygon = [s['geometry'] for s in collection if s['properties']['kv_id'] == int(project_id)][0]
            gt_date = [s["properties"]["PUB_DATO"] for s in collection if s['properties']['kv_id'] == int(project_id)][0]
            gt_date = parse_gt_date(gt_date)
            break
        except IndexError: pass 
    if polygon is None: raise ValueError(f"Could not find polygon and dates")
    return gt_date

class ExperimentVisualizer():
    def __init__(self, rcus, exp_var_name, exp_vars, variable_names, fig_ncols=2):
        self.exp_var_name = exp_var_name 
        self.exp_vars = exp_vars
        self.rcus = rcus
        self.variable_names = variable_names
        self.fig_ncols = fig_ncols
        self.df = self._make_df()

    def _make_df(self):
        for i, df in enumerate(self.rcus):
            self.rcus[i].results[self.exp_var_name] = [self.exp_vars[i] for _ in range(self.rcus[i].results.shape[0])]
        return pd.concat([r.results for r in self.rcus])

    @classmethod
    def from_paths(cls, paths, exp_var_name, exp_vars, variable_names, *args, **kwargs):
        assert isinstance(paths, list)
        rcus = [StratifiedRCU.from_json(os.path.join(p, "rcu.json")) for p in paths]
        return cls(rcus, exp_var_name, exp_vars, variable_names, *args, **kwargs)
    
    @classmethod
    def from_wandb(cls, *args, **kwargs):
        raise NotImplementedError()
    
    def variable_histogram_plot(self, variable, ax=None, hi_bound=np.inf, log=True, palette=None, show_legend=True):
        if ax is None:
            fig, ax = plt.subplots()
        var_idx = self.variable_names.index(variable)
        hdf = pd.DataFrame(columns=[self.exp_var_name, "probability", "variance"])
        for i, rcu in enumerate(self.rcus):
            H = np.nansum(rcu.histogram[0,var_idx], axis=0)
            ct = np.full(H.shape, str(self.exp_vars[i]))
            bins = rcu.histogram.bins[var_idx]
            hdf = pd.concat([hdf, pd.DataFrame({self.exp_var_name: ct, "probability": H/np.nansum(H), "variance": bins})])
            hdf = hdf[hdf.variance<hi_bound]
        hhdf = hdf.groupby(["variance", self.exp_var_name]).sum().reset_index()
        hhdf = hhdf.sort_values(by=[self.exp_var_name])
        if palette is not None: kwargs = {"palette": palette}
        else: kwargs = {}
        g = sns.lineplot(data=hhdf, x="variance", hue=self.exp_var_name, y="probability", ax=ax, alpha=0.5, **kwargs)
        if log: 
            ax.set_xscale("log")
            ax.set_xlabel("log variance")
        if not show_legend:
            ax.get_legend().remove()
        else:
            ax.legend(loc="upper right")
        return ax
    
    def histogram_plot(self, hi_bounds=np.inf, log=True, figsize=(12, 18), save_name=None, **kwargs):
        if not isinstance(hi_bounds, list): hi_bounds = [hi_bounds for _ in range(len(self.variable_names))]
        num_variables = len(self.variable_names)
        ncols = self.fig_ncols
        nrows = 1 if self.fig_ncols>=num_variables else math.ceil(num_variables/self.fig_ncols)
        fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
        axs = axs.flatten()
        for i, var in enumerate(self.variable_names):
            self.variable_histogram_plot(var, axs[i], hi_bounds[i], log, **kwargs)
            axs[i].set_title(var)
        if num_variables%self.fig_ncols!=0: fig.delaxes(axs.flatten()[-1])
        if save_name is not None: 
            fig.tight_layout()
            savefigure(fig, save_name)
        return axs

    def variable_metric_plot(self, metric, variable, kind, ax=None):
        if ax is None:
            fig, ax = plt.subplots()
        # plot global for each variable
        vmdf = self.df.query(f"kind == '{kind}'")
        vmdf = vmdf.query(f"metric == '{metric}' & variable == '{variable}'")
        gvmdf = vmdf.query("group=='global'")
        pvmdf = vmdf[~vmdf.group.isin(["global", "west", "north", "east"])]
        sns.scatterplot(data=pvmdf, x=self.exp_var_name, y="x", alpha=0.3, ax=ax)
        sns.lineplot(data=gvmdf, x=self.exp_var_name, y="x", ax=ax)
        ax.set_ylabel(metric)
        return ax

    def metric_plot(self, metric, kind, figsize=(12, 18), fig_ncols=None, save_name=None):
        num_variables = len(self.variable_names)
        if fig_ncols is not None:
            previous_ncols = self.fig_ncols
            self.fig_ncols = fig_ncols
        ncols = self.fig_ncols
        nrows = 1 if self.fig_ncols>=num_variables else math.ceil(num_variables/self.fig_ncols)
        fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
        axs = axs.flatten()
        for i, var in enumerate(self.variable_names):
            self.variable_metric_plot(metric, var, kind, axs[i])
            axs[i].set_title(var)
            axs[i].set_ylabel("")
        fig.suptitle(metric)
        if num_variables%self.fig_ncols!=0: fig.delaxes(axs.flatten()[-1])
        if fig_ncols is not None: self.fig_ncols = previous_ncols
        if save_name is not None: 
            fig.tight_layout()
            savefigure(fig, save_name)
        return axs

    def variable_metric_boxplot(self, metric, variable, kind, exp_var_bins=None, ax=None, fig_ncols=None):
        if ax is None:
            fig, ax = plt.subplots()
        # plot global for each variable
        vmdf = self.df.query(f"kind == '{kind}'").copy()
        vmdf = vmdf.query(f"metric == '{metric}' & variable == '{variable}'")
        gvmdf = vmdf.query("group=='global'")
        if exp_var_bins is None:
            sns.boxplot(data=gvmdf, y="x", ax=ax, showfliers=False)
        else:
            bin_strings = [f"{exp_var_bins[i]}-{exp_var_bins[i+1]}" for i in range(len(exp_var_bins)-1)]
            gvmdf.loc[:,self.exp_var_name] = [bin_strings[i-1] for i in np.digitize(gvmdf[self.exp_var_name].values, bins=exp_var_bins)]
            sns.boxplot(data=gvmdf, y="x", x=self.exp_var_name, ax=ax, order=bin_strings, showfliers=False)
        ax.set_ylabel(metric)
        return ax

    def metric_boxplot(self, metric, kind, exp_var_bins=None, figsize=(12, 18), fig_ncols=None, save_name=None):
        num_variables = len(self.variable_names)
        if fig_ncols is not None:
            previous_ncols = self.fig_ncols
            self.fig_ncols = fig_ncols
        ncols = self.fig_ncols
        nrows = 1 if self.fig_ncols>=num_variables else math.ceil(num_variables/self.fig_ncols)
        fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
        axs = axs.flatten()
        for i, var in enumerate(self.variable_names):
            self.variable_metric_boxplot(metric, var, kind, exp_var_bins, axs[i], fig_ncols)
            axs[i].set_title(var)
            axs[i].set_ylabel("")
        fig.suptitle(metric)
        if num_variables%self.fig_ncols!=0: fig.delaxes(axs.flatten()[-1])
        if fig_ncols is not None: self.fig_ncols = previous_ncols
        if save_name is not None: 
            fig.tight_layout()
            savefigure(fig, save_name)
        return axs
    
    def variable_calibration_plot(self, metric, variable, k=100, log_bins=False, ax=None, hi_bound=np.inf, palette=None, show_legend=True):
        bin_type = "log" if log_bins else "linear"
        if ax is None:
            fig, ax = plt.subplots()
        var_idx = self.variable_names.index(variable)
        assert metric in ["uce", "ence", "auce"]
        if metric == "uce": cols = ["bin variance", "bin mse", self.exp_var_name]
        elif metric == "auce": cols = ["expected accuracy", "empirical accuracy", self.exp_var_name]
        else: cols = ["bin std", "bin rmse", self.exp_var_name]
        ccdf = pd.DataFrame(columns=cols)
        for i, rcu in enumerate(self.rcus):
            lorcu = rcu.upsample(k, bin_type=bin_type)
            xc, yc = lorcu.get_calibration_curve(metric)
            if metric != "auce": xc = xc[var_idx]
            ct = np.full(xc.shape, str(self.exp_vars[i]))
            ccdf = pd.concat([ccdf, pd.DataFrame({cols[0]:xc, cols[1]:yc[var_idx], cols[-1]:ct[var_idx]})])
        ccdf = ccdf.dropna()
        ccdf = ccdf.sort_values(by=[self.exp_var_name])
        lo = min(ccdf[cols[0]].min(), ccdf[cols[1]].min())
        hi = min(hi_bound, max(ccdf[cols[0]].max(), ccdf[cols[1]].max()))
        id_line = (
            np.linspace(lo, hi, 2), 
            np.linspace(lo, hi, 2)
        )
        ccdf = ccdf[ccdf[cols[0]]<hi_bound]
        ax.plot(*id_line, color="black", linestyle="dotted")
        if palette is not None: kwargs = {"palette": palette}
        else: kwargs = {}
        sns.lineplot(data=ccdf, x=cols[0], y=cols[1], hue=self.exp_var_name, ax=ax, alpha=0.5, **kwargs)
        if not show_legend:
            ax.get_legend().remove()
        else:
            ax.legend(loc="upper right")
        # ax.vlines(x=xc, ymin=lo, ymax=hi/5, color="black", ls="dashed")
        return ax
    
    def calibration_plot(self, metric, k=100, log_bins=False, hi_bounds=np.inf, figsize=(12, 18), fig_ncols=None, save_name=None, **kwargs):
        if not isinstance(hi_bounds, list): hi_bounds = [hi_bounds for _ in range(len(self.variable_names))]
        num_variables = len(self.variable_names)
        if fig_ncols is not None:
            previous_ncols = self.fig_ncols
            self.fig_ncols = fig_ncols
        ncols = self.fig_ncols
        nrows = 1 if self.fig_ncols>=num_variables else math.ceil(num_variables/self.fig_ncols)
        fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
        axs = axs.flatten()
        for i, var in enumerate(self.variable_names):
            self.variable_calibration_plot(metric, var, k=k, log_bins=log_bins, ax=axs[i], hi_bound=hi_bounds[i], **kwargs)
            axs[i].set_title(var)
        fig.suptitle(metric+" calibration plot")
        if num_variables%self.fig_ncols!=0: fig.delaxes(axs.flatten()[-1])
        if fig_ncols is not None: self.fig_ncols = previous_ncols
        if save_name is not None: 
            fig.tight_layout()
            savefigure(fig, save_name)
        return axs

def getPaths(
    src_dir, 
    s2repr_dirs=None, 
    gt_dir=None,
    returns=None
):
    src_dir = str(src_dir)
    if returns is None:
        return
    rname = lambda x: "_".join(Path(x).stem.split("_")[1:])
    valid_returns = ["img", "gt"] + [rname(p.path) for p in os.scandir(src_dir)]
    # start logic
    dir_name = src_dir.split("/")[-1]
    pid = dir_name.split("_")[0]
    retuple = []
    for r in returns: 
        assert r in valid_returns
        if r=="img":
            assert s2repr_dirs is not None, "must provide s2repr_dirs to get img"
            retuple.append(
                list(Path(os.path.join(s2repr_dirs, dir_name, pid)).glob("*.tif"))[0]
            )
        elif r=="gt":
            assert gt_dir is not None, "must provide gt_dir to get gt"
            retuple.append(os.path.join(gt_dir, f"{pid}.tif"))
        else:
            retuple.append(os.path.join(src_dir, f"{pid}_{r}.tif"))
        assert os.path.exists(retuple[-1])
    if len(returns) > 1: return tuple(retuple)
    else: return retuple[0]

def loadRaster(
    path,
    bands=None, # set to None for all bands, -1 for last one
    islice=None,
    jslice=None,
    clip_range=None,
    transpose_order=None,
    set_nan_mask=False, # for gt
    dtype=None,
    elementwise_fn=None
):
    with rasterio.open(path) as f:
        if bands is None: bands = f.indexes
        if bands == -1: bands = f.count
        if isinstance(bands, tuple): bands = list(bands)
        x = f.read(bands)
        if set_nan_mask:
            mask = f.read_masks(1)
            if isinstance(bands, list): x[:,mask==0] = np.nan
            else: x[mask==0] = np.nan
        if islice is not None:
            if isinstance(bands, list): x = x[:,islice]
            else: x = x[islice]
        if jslice is not None:
            if isinstance(bands, list): x = x[:,:,jslice]
            else: x = x[:,jslice]
        if clip_range is not None: x = clip(x, clip_range)
        if transpose_order is not None: x = x.transpose(*transpose_order)
        if dtype is not None: x = x.astype(dtype)
        if elementwise_fn is not None: x = elementwise_fn(x)
    return x
    
def clip(arr, bounds):
    bounds = (float(bounds[0]), float(bounds[1]))
    arr = np.where(arr>bounds[1], bounds[1], arr)
    arr = np.where(arr<bounds[0], bounds[0], arr)
    arr -= bounds[0]
    arr /= (bounds[1]-bounds[0])
    return arr

def multiMinMax(*sources, axis=None):
    return min(*[np.nanmin(source, axis=axis) for source in sources]), max(*[np.nanmax(source, axis=axis) for source in sources])

def norm2d(x, mn=None, mx=None, a=0, b=1): 
    if mn is None: mn = np.nanmin(x)
    if mx is None: mx = np.nanmax(x)
    return (b-a)*(x-mn)/(mx-mn)+a

def multiNorm2d(*sources, a=0, b=1, axis=None, return_bounds=False):
    mn, mx = multiMinMax(*sources, axis)
    res = [norm2d(source, mn, mx, a, b) for source in sources]
    if return_bounds: return tuple([res]+[min, mx])
    else: return tuple(res)

def savefigure(fig, name):
    Path(name).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(name+".png", dpi=300)
    fig.savefig(name+".pdf", dpi=200)

def filterZeroAvgCp(result_dirs, s2repr_dirs):
    exp_vars =  []
    nnz_result_dirs = []
    for result_dir in result_dirs:
        img_path = getPaths(result_dir, s2repr_dirs, returns=["img"])
        mean = loadRaster(img_path, bands=-1, dtype="float").mean()
        if mean != 0. and mean != 100.:
            exp_vars.append(mean)
            nnz_result_dirs.append(result_dir)
    return exp_vars, nnz_result_dirs

def get_nonzero_avg_cp_visualizer(result_dirs, s2repr_dirs, variable_names, max_n=None, bins=None):
    exp_vars, nnz_result_dirs = filterZeroAvgCp(result_dirs, s2repr_dirs)
    if bins is not None:
        bin_ids = np.digitize(exp_vars, bins=bins)-1
        selected_exp_vars, selected_nnz_result_dirs = [], []
        for bin_id in np.unique(bin_ids):
            bin_exp_vars = [exp_vars[i] for i in range(len(exp_vars)) if bin_ids[i]==bin_id]
            bin_nnz_result_dirs = [nnz_result_dirs[i] for i in range(len(nnz_result_dirs)) if bin_ids[i]==bin_id]
            assert len(bin_exp_vars)==len(bin_nnz_result_dirs)
            print(f"bin {bin_id}: {len(bin_exp_vars)} results")
            if max_n is not None:
                selected_exp_vars.extend(bin_exp_vars[:max_n//(len(bins)-1)])
                selected_nnz_result_dirs.extend(bin_nnz_result_dirs[:max_n//(len(bins)-1)])
            else:
                selected_exp_vars.extend(bin_exp_vars)
                selected_nnz_result_dirs.extend(bin_nnz_result_dirs)
    else:
        selected_exp_vars, selected_nnz_result_dirs = exp_vars, nnz_result_dirs
    print(f"selected {len(selected_exp_vars)} results.")
    # Create visualizer object
    visualizer = ExperimentVisualizer.from_paths(
        selected_nnz_result_dirs,
        "average cloud probability",
        selected_exp_vars, 
        variable_names,
        fig_ncols=2
    )
    return visualizer, selected_nnz_result_dirs, selected_exp_vars

def showSplit(gt_path, split_mask, islices, jslices, 
              patch_colors=["r", "b"], save_name=None):
    reverse_split_mask = split_mask.copy()
    reverse_split_mask[split_mask==0] = 2
    reverse_split_mask[split_mask==2] = 0
    split_mask = reverse_split_mask
    fig = plt.figure(figsize=(5,4))
    with rasterio.open(gt_path) as f:
        gts = []
        gt_mask = f.read_masks(1)//255
        for i in range(1,f.count+1):
            gt = f.read(i)
            gt[gt_mask==0] = np.nan
            gt[gt>0] = 100
            gts.append(gt)
        gt = np.nanmean(np.stack(gts, axis=0), axis=0)
    g = sns.heatmap(split_mask, 
                    cmap=sns.color_palette("gist_earth_r", 3), 
                    alpha=0.3)
    g = sns.heatmap(gt, cmap="Greys", ax=g, cbar=False)
    # g = sns.heatmap(gt, cmap="Greys", cbar=False)
    assert len(islices)==len(jslices)==len(patch_colors)
    for i, (islc, jslc) in enumerate(zip(islices, jslices)):
        g.add_patch(
            patches.Rectangle(
                (jslc.start, islc.start), # top left corner
                jslc.stop-jslc.start, # positive width
                islc.stop-islc.start, # positive height
                linewidth=2,
                fill=False,
                color=patch_colors[i]
            )
        )
    g.set_axis_off()
    # modify colorbar:
    colorbar = g.collections[0].colorbar 
    r = colorbar.vmax - colorbar.vmin 
    colorbar.set_ticks([colorbar.vmin + r / 3 * (0.5 + i) for i in range(3)])
    colorbar.ax.set_yticklabels(["test", "validation", "train"])   
    plt.tight_layout()
    if save_name is not None:
        fig.savefig(f"{save_name}.png", dpi=300)
        fig.savefig(f"{save_name}.pdf", dpi=200)
    plt.show()

def showRGB(dirs, s2repr_dirs, titles=None, islice=None, jslice=None, draw_bbox=False, 
            figsize=(12, 4), split_mask=None, save_name=None, color="g", nrows=1):
    n = len(dirs)
    fig, axs = plt.subplots(ncols=math.ceil(len(dirs)/nrows), nrows=nrows, figsize=figsize)
    if len(dirs)>1 or nrows>1: axs = axs.flatten()
    else: axs = [axs]
    i = 0
    for d, ax in zip(dirs, axs):
        dir_name = d.split("/")[-1]
        pid = dir_name.split("_")[0]
        img_path = list(Path(os.path.join(s2repr_dirs, dir_name, pid)).glob("*.tif"))[0]
        title = i if titles is None else titles[i]
        rgb = loadRaster(img_path, bands=[4,3,2], clip_range=(100, 2000), transpose_order=(1,2,0))
        if islice is not None and jslice is not None: 
            if not isinstance(islice, list): islice = [islice]
            if not isinstance(jslice, list): jslice = [jslice]
            if isinstance(color, list): assert len(color)==len(islice)
            assert len(islice)==len(jslice)
            for k, (islc, jslc) in enumerate(zip(islice, jslice)):
                if not draw_bbox:
                    rgb = rgb[islc,jslc]
                    ax.imshow(rgb)
                    if split_mask is not None:
                        ax.imshow(split_mask[islc, jslc], alpha=0.2)
                else:
                    ax.imshow(rgb)
                    if split_mask is not None:
                        ax.imshow(split_mask, alpha=0.2)
                    if isinstance(color, list): kw = {"color": color[k]}
                    elif isinstance(color, str): kw = {"color": color}
                    else: kw={}
                    ax.add_patch(
                        patches.Rectangle(
                            (jslc.start, islc.start), # top left corner
                            jslc.stop-jslc.start, # positive width
                            islc.stop-islc.start, # positive height
                            linewidth=2,
                            fill=False,
                            **kw
                        )
                    )
        else:
            ax.imshow(rgb)
        ax.set_title(title)
        ax.set_axis_off()
        i += 1
    plt.tight_layout()
    if save_name is not None: savefigure(fig, save_name)
    plt.show()

def showPairedMaps(matching_dirs, variable_index, variable_name, islice=None, jslice=None, 
                     normalize=False, save_name=None, figsize=(15,11)):
    fig, axs = plt.subplots(nrows=len(matching_dirs), ncols=6, figsize=figsize)
    for i, (orig_dir, gee_dir) in enumerate(matching_dirs):
        # paths
        gee_mean_path, gee_variance_path = getPaths(gee_dir, returns=["mean", "variance"])
        orig_mean_path, orig_variance_path = getPaths(orig_dir, returns=["mean", "variance"])
        # means
        gee_mean = loadRaster(gee_mean_path, bands=variable_index, islice=islice, jslice=jslice)
        orig_mean = loadRaster(orig_mean_path, bands=variable_index, islice=islice, jslice=jslice)
        # variances
        gee_predictive_std = loadRaster(gee_variance_path, bands=variable_index, islice=islice, jslice=jslice, elementwise_fn=np.sqrt)
        orig_predictive_std = loadRaster(orig_variance_path, bands=variable_index, islice=islice, jslice=jslice, elementwise_fn=np.sqrt)
        if normalize:
            # # get stats
            # meanmax, meanmin = np.nanmax(orig_mean), np.nanmin(orig_mean)
            # variancemax, variancemin = np.nanmax(orig_variance), np.nanmin(orig_variance)
            # # apply
            # gee_mean = norm2d(gee_mean, meanmin, meanmax)
            # orig_mean = norm2d(orig_mean, meanmin, meanmax)
            # gee_variance = norm2d(gee_variance, variancemin, variancemax)
            # orig_variance = norm2d(orig_variance, variancemin, variancemax)
            orig_mean, gee_mean, orig_predictive_std, gee_predictive_std = multiNorm2d(orig_mean, gee_mean, orig_predictive_std, gee_predictive_std)
            vmin, vmax = 0, 1
        else:
            vmin, vmax = multiMinMax(orig_mean, gee_mean, orig_predictive_std, gee_predictive_std)
        dvmin, dvmax = vmin-vmax, vmax-vmin
        meandiff = gee_mean-orig_mean
        predictive_stddiff = gee_predictive_std-orig_predictive_std
        mbound = np.nanmax(np.abs(meandiff))
        pbound = np.nanmax(np.abs(predictive_stddiff))
        sns.heatmap(orig_mean, ax=axs[i,0], vmin=vmin, vmax=vmax)
        if i==0: axs[i,0].set_title(f"original mean")
        sns.heatmap(gee_mean, ax=axs[i,1], vmin=vmin, vmax=vmax)
        if i==0: axs[i,1].set_title(f"gee mean")
        sns.heatmap(meandiff, ax=axs[i,2], cmap="bwr", vmin=-mbound, vmax=mbound)
        if i==0: axs[i,2].set_title(f"difference mean")
        sns.heatmap(orig_predictive_std, ax=axs[i,3], vmin=vmin, vmax=vmax)
        if i == 0: axs[i,3].set_title(f"original PU")
        sns.heatmap(gee_predictive_std, ax=axs[i,4], vmin=vmin, vmax=vmax)
        if i == 0: axs[i,4].set_title(f"gee PU")
        sns.heatmap(predictive_stddiff, ax=axs[i,5], cmap="bwr", vmin=-pbound, vmax=pbound)
        if i == 0: axs[i,5].set_title(f"difference PU")
    for ax in axs.flatten(): 
        ax.set_xticks([])
        ax.set_yticks([])
    for i, (d, _) in enumerate(matching_dirs):
        axs[i,0].set_ylabel(datetime.strptime(Path(d).name.split("_")[1].split("T")[0], '%Y%m%d').strftime("%d.%m.%Y"))
    fig.suptitle(variable_name)
    plt.tight_layout()
    if save_name is not None: savefigure(fig, save_name)
    plt.show()
    
def showPairedHistograms(matching_dirs, variable_index, variable_name, islice=None, jslice=None, 
                           log_mean=False, log_uncertainty=False, save_name=None, figsize=(10, 15),
                           normalize=False):
    fig, axs = plt.subplots(nrows=len(matching_dirs), ncols=2, figsize=figsize)
    for i, (orig_dir, gee_dir) in enumerate(matching_dirs):
        # paths
        gee_mean_path, gee_variance_path = getPaths(gee_dir, returns=["mean", "variance"])
        orig_mean_path, orig_variance_path = getPaths(orig_dir, returns=["mean", "variance"])
        # means
        gee_mean = loadRaster(gee_mean_path, bands=variable_index, islice=islice, jslice=jslice).flatten()
        orig_mean = loadRaster(orig_mean_path, bands=variable_index, islice=islice, jslice=jslice).flatten()
        # variances
        gee_predictive_std = loadRaster(gee_variance_path, bands=variable_index, islice=islice, jslice=jslice, elementwise_fn=np.sqrt).flatten()
        orig_predictive_std = loadRaster(orig_variance_path, bands=variable_index, islice=islice, jslice=jslice, elementwise_fn=np.sqrt).flatten()
        # normalize
        if normalize:
            orig_mean, gee_mean, orig_predictive_std, gee_predictive_std = multiNorm2d(orig_mean, gee_mean, orig_predictive_std, gee_predictive_std)
        km = "mean" if not log_mean else "log mean"
        mdf = pd.DataFrame({
            "source": ["original" for _ in orig_mean]+["gee" for _ in gee_mean],
            km: orig_mean.tolist()+gee_mean.tolist()
        })
        kv = "predictive uncertainty" if not log_uncertainty else "log predictive uncertainty"
        vdf = pd.DataFrame({
            "source": ["original" for _ in orig_predictive_std]+["gee" for _ in gee_predictive_std],
            kv: orig_predictive_std.tolist()+gee_predictive_std.tolist()
        })
        sns.histplot(data=mdf, x=km, hue="source", ax=axs[i,0], multiple="dodge", bins=20)
        axs[i,0].set_title(datetime.strptime(Path(orig_dir).name.split("_")[1].split("T")[0], '%Y%m%d').strftime("%d.%m.%Y"))
        if log_mean: axs[i,0].set_yscale("log")
        sns.histplot(data=vdf, x=kv, hue="source", ax=axs[i,1], multiple="dodge", bins=20)
        axs[i,1].set_title(datetime.strptime(Path(orig_dir).name.split("_")[1].split("T")[0], '%Y%m%d').strftime("%d.%m.%Y"))
        axs[i,1].set_ylabel("")
        if log_uncertainty: axs[i,1].set_yscale("log")
        axs[i,0].get_legend().remove()
        axs[i,1].get_legend().remove()
    fig.suptitle(variable_name)
    fig.legend([axs[0,0], axs[0,1]], labels=["original", "gee"], ncol=2, loc='upper center', bbox_to_anchor=(0.5, 0.))
    plt.tight_layout()
    if save_name is not None: savefigure(fig, save_name)
    plt.show()

def showMeanDifferenceMaps(orig_path, gee_path, 
                           variable_index, 
                           variable_name,
                           islices, jslices,
                           row_labels,
                           normalize=False, 
                           save_name=None,
                           nrows=2, 
                           figsize=(15,10)):
    assert len(islices)==len(jslices)
    assert len(islices)%nrows==0
    gee_dir = Path(gee_path).parent
    # Load data
    orig_mean = loadRaster(orig_path, bands=variable_index)
    gee_mean = loadRaster(gee_path, bands=variable_index)
    # figure
    fig, axs = plt.subplots(nrows=nrows, ncols=len(islices)//nrows, figsize=figsize)
    axs = axs.flatten()
    colors = ["r" for _ in range(len(islices)//nrows+1)]+\
             ["orange" for _ in range(len(islices)//nrows+1)]
    j = 0
    for i, (ax, islice, jslice, color) in enumerate(
        zip(axs.flatten(), islices, jslices, colors)
    ):
        O = orig_mean[islice,jslice].copy()
        G = gee_mean[islice,jslice].copy()
        if normalize:
            O, G = multiNorm2d(O, G)
            vmin, vmax = 0, 1
        else:
            vmin, vmax = multiMinMax(O, G)
        dvmin, dvmax = vmin-vmax, vmax-vmin
        rerror = G-O
        rbound = np.nanmax(np.abs(rerror))
        sns.heatmap(
            rerror,
            cmap="bwr",
            vmin=-rbound, 
            vmax=rbound,
            ax=ax
        )
        ax.set_xticks([])
        ax.set_yticks([])
    axs = axs.reshape(nrows, -1)
    for i in range(nrows):
        axs[i, 0].set_ylabel(row_labels[i])
    fig.suptitle(f'{variable_name} ({datetime.strptime(Path(gee_dir).name.split("_")[1].split("T")[0], "%Y%m%d").strftime("%d.%m.%Y")})')
    plt.tight_layout()
    if save_name is not None: savefigure(fig, save_name)
    plt.show()

def precomputeCbarBoundsUncetaintyTypes(
    dirs, s2repr_dirs, variable_names, islice=None, jslice=None, quantile=90,
):
    std_bounds = ([np.inf for _ in variable_names], [-np.inf for _ in variable_names])
    epistemic_diff_bounds = [-np.inf for _ in variable_names]
    aleatoric_diff_bounds = [-np.inf for _ in variable_names]
    first_aleatoric, first_epistemic = None, None
    for i, d in enumerate(dirs):
        aleatoric_path, epistemic_path = getPaths(d, s2repr_dirs=s2repr_dirs, returns=["aleatoric", "epistemic"])
        aleatoric = loadRaster(aleatoric_path, bands=None, islice=islice, jslice=jslice, elementwise_fn=np.sqrt)
        epistemic = loadRaster(epistemic_path, bands=None, islice=islice, jslice=jslice, elementwise_fn=np.sqrt)
        predictive = aleatoric + epistemic
        if i==0:
            aleadiff = aleatoric-aleatoric
            first_aleatoric = aleatoric.copy()
            epidiff = epistemic-epistemic
            first_epistemic = epistemic.copy()
        else:
            aleadiff = aleatoric - first_aleatoric
            epidiff = epistemic - first_epistemic
        for i, _ in enumerate(variable_names):
            pmn, pmx = np.nanmin(predictive[i]), np.nanmax(predictive[i])
            ab, eb = np.nanpercentile(np.abs(aleadiff), q=quantile), np.nanpercentile(np.abs(epidiff), q=quantile)
            if pmn < std_bounds[0][i]: std_bounds[0][i] = pmn
            if pmx > std_bounds[1][i]: std_bounds[1][i] = pmx
            if ab > aleatoric_diff_bounds[i]: aleatoric_diff_bounds[i] = ab
            if eb > epistemic_diff_bounds[i]: epistemic_diff_bounds[i] = eb
    return std_bounds, aleatoric_diff_bounds, epistemic_diff_bounds

def showUncertaintyTypes(dirs, titles, variable_index, variable_name, s2repr_dirs,
                         islice=None, jslice=None, normalize=False, 
                         std_bounds=None, epistemic_diff_bounds=None, aleatoric_diff_bounds=None,
                         save_name=None, figsize=(15,7.5)):
    fig, axs = plt.subplots(nrows=len(titles), ncols=6, figsize=figsize)
    first_aleatoric_std, first_epistemic_std = None, None
    for i, d in enumerate(dirs):
        img_path, variance_path, aleatoric_path, epistemic_path = getPaths(
            d, s2repr_dirs, 
            returns=["img","variance","aleatoric", "epistemic"]
        )
        # load rasters
        rgb = loadRaster(img_path, bands=[4,3,2], transpose_order=(1,2,0), clip_range=(100, 2000), islice=islice, jslice=jslice)
        predictive_std = loadRaster(variance_path, bands=variable_index, islice=islice, jslice=jslice, elementwise_fn=np.sqrt)
        aleatoric_std = loadRaster(aleatoric_path, bands=variable_index, islice=islice, jslice=jslice, elementwise_fn=np.sqrt)
        epistemic_std = loadRaster(epistemic_path, bands=variable_index, islice=islice, jslice=jslice, elementwise_fn=np.sqrt)
        if i==0: 
            first_aleatoric_std = aleatoric_std
            first_epistemic_std = epistemic_std
        adiff = aleatoric_std - first_aleatoric_std
        ediff = epistemic_std - first_epistemic_std
        if normalize:
            predictive_std, aleatoric_std, epistemic_std = multiNorm2d(predictive_std, aleatoric_std, epistemic_std)
            vmin, vmax = 0, 1
        else:
            vmin, vmax = multiMinMax(predictive_std, aleatoric_std, epistemic_std)
        if std_bounds is None:
            stdmin, stdmax = vmin, vmax
        else:
            stdmin, stdmax = std_bounds[0][variable_index-1], std_bounds[1][variable_index-1]        
        adbound = aleatoric_diff_bounds[variable_index-1] if aleatoric_diff_bounds is not None else vmax
        edbound = epistemic_diff_bounds[variable_index-1] if epistemic_diff_bounds is not None else vmax
        axs[i,0].imshow(rgb)
        if i ==0: axs[i,0].set_title(f"RGB")
        sns.heatmap(epistemic_std, ax=axs[i,1], vmin=stdmin, vmax=stdmax)
        if i==0: axs[i,1].set_title(f"epistemic")
        sns.heatmap(aleatoric_std, ax=axs[i,2], vmin=stdmin, vmax=stdmax)
        if i==0: axs[i,2].set_title(f"aleatoric")
        sns.heatmap(predictive_std, ax=axs[i,3], vmin=stdmin, vmax=stdmax)
        if i==0: axs[i,3].set_title(f"predictive")
        sns.heatmap(ediff, ax=axs[i,4], cmap="bwr", vmin=-edbound, vmax=edbound)
        if i==0: axs[i,4].set_title(f"epistemic difference")
        sns.heatmap(adiff, ax=axs[i,5], cmap="bwr", vmin=-adbound, vmax=adbound)
        if i==0: axs[i,5].set_title(f"aleatoric difference")
    for ax in axs.flatten(): 
        ax.set_xticks([])
        ax.set_yticks([])
    for i, d, in enumerate(dirs):
        axs[i,0].set_ylabel(f'{titles[i]}')
    fig.suptitle(variable_name)
    plt.tight_layout()
    if save_name is not None: savefigure(fig, save_name)
    plt.show()

def precomputeCbarBoundsPredictionMaps(
    dirs,
    s2repr_dirs,
    gt_dir,
    variable_names,
    islice=None,
    jslice=None,
    quantile=75,
):
    # initialize
    predictive_uncertainty_bounds = ([None for _ in variable_names], [None for _ in variable_names])
    rerror_bounds = [None for _ in variable_names]
    cerror_bounds = [None for _ in variable_names]
    # loop on dirs
    for j, d in enumerate(dirs):
        mean_path, variance_path, gt_path = getPaths(
            d, gt_dir=gt_dir, s2repr_dirs=s2repr_dirs, returns=["mean", "variance", "gt"]
        )
        gt = loadRaster(gt_path, bands=None, islice=islice, jslice=jslice)
        mean = loadRaster(mean_path, bands=None, islice=islice, jslice=jslice)
        predictive_std = loadRaster(variance_path, bands=None, islice=islice, jslice=jslice, elementwise_fn=np.sqrt)
        absrerror = np.abs(mean-gt)
        abscerror = np.abs(absrerror-predictive_std)
        # update bounds
        for i, _ in enumerate(variable_names):
            pb0, pb1 = np.nanmin(predictive_std[i]), np.nanmax(predictive_std[i])
            rb, cb = np.nanpercentile(absrerror, q=quantile), np.nanpercentile(abscerror, q=quantile)
            if j==0 or pb0 < predictive_uncertainty_bounds[0][i]: predictive_uncertainty_bounds[0][i] = pb0
            if j==0 or pb1 > predictive_uncertainty_bounds[1][i]: predictive_uncertainty_bounds[1][i] = pb1
            if j==0 or rb > rerror_bounds[i]: rerror_bounds[i] = rb
            if j==0 or cb > cerror_bounds[i]: cerror_bounds[i] = cb
    return predictive_uncertainty_bounds, rerror_bounds, cerror_bounds
    
def showPredictionMaps(dirs, titles, variable_index, variable_name, s2repr_dirs, gt_dir, 
                        shapefile_paths, islice=None, jslice=None, normalize=False, save_name=None,
                        predictive_uncertainty_bounds=None, rerror_bounds=None, cerror_bounds=None,
                        figsize=(15,7.5)):
    fig, axs = plt.subplots(nrows=len(titles), ncols=6, figsize=figsize)
    for i, d in enumerate(dirs):
        dir_name = d.split("/")[-1]
        pid = dir_name.split("_")[0]
        gt_date = compute_gt_date(pid, shapefile_paths)
        img_path, mean_path, variance_path, gt_path = getPaths(d, s2repr_dirs, gt_dir, returns=["img","mean","variance","gt"])
        # load rasters
        rgb = loadRaster(img_path, bands=[4,3,2], transpose_order=(1,2,0), clip_range=(100, 2000), islice=islice, jslice=jslice)
        gt = loadRaster(gt_path, bands=variable_index, islice=islice, jslice=jslice, set_nan_mask=True)
        if variable_index in [3,5]: gt /= 100 # Cover/Dens normalization!!
        mean = loadRaster(mean_path, bands=variable_index, islice=islice, jslice=jslice)
        predictive_std = loadRaster(variance_path, bands=variable_index, islice=islice, jslice=jslice, elementwise_fn=np.sqrt)
        # if normalize:
        #     # get stats
        #     rgbmax, rgbmin = np.nanmax(rgb, axis=(0,1)), np.nanmin(rgb, axis=(0,1))
        #     gtmax, gtmin = np.nanmax(gt), np.nanmin(gt)
        #     variancemax, variancemin = np.nanmax(variance), np.nanmin(variance)
        #     # apply
        #     rgb = norm2d(rgb, rgbmin, rgbmax)
        #     gt = norm2d(gt, gtmin, gtmax)
        #     mean = norm2d(mean, gtmin, gtmax)
        #     # variance = norm2d(variance, variancemin, variancemax)
        #     variance = norm2d(variance, gtmin**2, gtmax**2)
        if normalize:
            gt, mean, predictive_std = multiNorm2d(gt, mean, predictive_std)
            vmin, vmax = 0, 1
        else:
            vmin, vmax = multiMinMax(gt, mean, predictive_std)
        dvmin, dvmax = vmin-vmax, vmax-vmin
        rerror = mean-gt
        cerror = np.abs(rerror)-predictive_std
        pubounds = (vmin, vmax) if predictive_uncertainty_bounds is None else (predictive_uncertainty_bounds[0][variable_index-1], predictive_uncertainty_bounds[1][variable_index-1])
        rbound = np.nanmax(np.abs(rerror)) if rerror_bounds is None else rerror_bounds[variable_index-1]
        cbound = np.nanmax(np.abs(cerror)) if cerror_bounds is None else cerror_bounds[variable_index-1]
        axs[i,0].imshow(rgb)
        if i ==0: axs[i,0].set_title(f"RGB")
        sns.heatmap(gt, ax=axs[i,1], vmin=vmin, vmax=vmax)
        if i==0: axs[i,1].set_title(f"ground truth\n({gt_date.strftime('%d.%m.%Y')})")
        sns.heatmap(mean, ax=axs[i,2], vmin=vmin, vmax=vmax)
        if i==0: axs[i,2].set_title(f"mean")
        sns.heatmap(predictive_std, ax=axs[i,3], vmin=pubounds[0], vmax=pubounds[1])
        if i==0: axs[i,3].set_title(f"predictive\nuncertainty")
        sns.heatmap(rerror, ax=axs[i,4], cmap="bwr", vmin=-rbound, vmax=rbound)
        if i==0: axs[i,4].set_title(f"r-error")
        sns.heatmap(cerror, ax=axs[i,5], cmap="bwr", vmin=-cbound, vmax=cbound)
        if i==0: axs[i,5].set_title(f"c-error")
    for ax in axs.flatten(): 
        ax.set_xticks([])
        ax.set_yticks([])
    for i, d, in enumerate(dirs):
        axs[i,0].set_ylabel(f'{titles[i]}\n({datetime.strptime(Path(d).name.split("_")[1].split("T")[0], "%Y%m%d").strftime("%d.%m.%Y")})')
    fig.suptitle(variable_name)
    plt.tight_layout()
    if save_name is not None: savefigure(fig, save_name)
    plt.show()

def showPredictionHistograms(dirs, titles, variable_index, variable_name,
                           gt_dir, shapefile_paths, log_mean=False, log_uncertainty=False,
                           islice=None, jslice=None, normalize=False,
                           trainset_means=None,
                           save_name=None, figsize=(10, 15)):
    fig, axs = plt.subplots(nrows=len(titles), ncols=2, figsize=figsize)
    for i, d in enumerate(dirs):
        dir_name = d.split("/")[-1]
        pid = dir_name.split("_")[0]
        gt_date = compute_gt_date(pid, shapefile_paths)
        mean_path, variance_path, gt_path = getPaths(d, gt_dir=gt_dir, returns=["mean","variance","gt"])
        # load rasters
        gt = loadRaster(gt_path, bands=variable_index, islice=islice, jslice=jslice, set_nan_mask=True)
        if variable_index in [3,5]: gt /= 100 # Cover/Dens normalization!!
        mean = loadRaster(mean_path, bands=variable_index, islice=islice, jslice=jslice)
        predictive_std = loadRaster(variance_path, bands=variable_index, islice=islice, jslice=jslice, elementwise_fn=np.sqrt)
        if normalize:
            gt, mean, predictive_std = multiNorm2d(gt, mean, predictive_std)
        km = "mean"
        if trainset_means is None:
            gt_title = f"gt ({gt_date.strftime('%d.%m.%Y')})"
            pred_title = f'prediction ({datetime.strptime(Path(d).name.split("_")[1].split("T")[0], "%Y%m%d").strftime("%d.%m.%Y")})'
        else:
            gt_title = f"gt (mean={trainset_means[variable_index-1]:.3f})"
            pred_title = f'prediction (mean={np.nanmean(mean):.3f})'
        mdf = pd.DataFrame({
            "source": [gt_title for _ in gt.flatten()]+
                      [pred_title for _ in mean.flatten()],
            km: gt.flatten().tolist()+mean.flatten().tolist()
        })
        kv = "predictive uncertainty"
        vdf = pd.DataFrame({
            "source": ["prediction" for _ in predictive_std.flatten()],
            kv: predictive_std.flatten().tolist()
        })
        axs[i,0] = sns.histplot(data=mdf, x=km, hue="source", ax=axs[i,0], multiple="dodge", bins=20)
        sns.move_legend(axs[i,0], "upper center", ncol=2, title="", fontsize="small", bbox_to_anchor=(0.5,1.15))
        if log_mean: axs[i,0].set_yscale("log")
        sns.histplot(data=vdf, x=kv, ax=axs[i,1], multiple="dodge", bins=20)
        # axs[i,1].set_title(titles[i])
        axs[i,1].set_ylabel("")
        if log_uncertainty: axs[i,1].set_yscale("log")
    for i, d, in enumerate(dirs):
        axs[i,0].set_ylabel(f'{titles[i]}\nCount')
    fig.suptitle(variable_name)
    plt.tight_layout()
    if save_name is not None: savefigure(fig, save_name)
    plt.show()

def showErrorHistograms(dirs, titles, variable_index, variable_name,
                           gt_dir, log_rerror=False, log_cerror=False, 
                           islice=None, jslice=None, normalize=False,
                           save_name=None, figsize=(10,15)):
    fig, axs = plt.subplots(nrows=len(titles), ncols=2, figsize=figsize)
    for i, d in enumerate(dirs):
        mean_path, variance_path, gt_path = getPaths(d, gt_dir=gt_dir, returns=["mean","variance","gt"])
        # load rasters
        gt = loadRaster(gt_path, bands=variable_index, islice=islice, jslice=jslice, set_nan_mask=True)
        if variable_index in [3,5]: gt /= 100 # Cover/Dens normalization!!
        mean = loadRaster(mean_path, bands=variable_index, islice=islice, jslice=jslice)
        predictive_std = loadRaster(variance_path, bands=variable_index, islice=islice, jslice=jslice, elementwise_fn=np.sqrt)
        if normalize:
            gt, mean, predictive_std = multiNorm2d(gt, mean, predictive_std)
        rerror = mean-gt
        cerror = np.abs(rerror)-predictive_std
        rk = "r-error" if not log_rerror else "log r-error"
        rdf = pd.DataFrame({
            "source": ["prediction" for _ in rerror.flatten()],
            rk: rerror.flatten().tolist()
        })
        ck = "c-error" if not log_cerror else "log c-error"
        cdf = pd.DataFrame({
            "source": ["prediction" for _ in cerror.flatten()],
            ck: cerror.flatten().tolist()
        })
        sns.histplot(data=rdf, x=rk, ax=axs[i,0], multiple="dodge", bins=20)
        # axs[i,0].set_title(titles[i])
        if log_rerror: axs[i,0].set_xscale("log")
        sns.histplot(data=cdf, x=ck, ax=axs[i,1], multiple="dodge", bins=20)
        # axs[i,1].set_title(titles[i])
        axs[i,1].set_ylabel("")
        if log_cerror: axs[i,1].set_xscale("log")
    for i, d, in enumerate(dirs):
        axs[i,0].set_ylabel(f'{titles[i]}\nCount')
    fig.suptitle(variable_name)
    plt.tight_layout()
    if save_name is not None: savefigure(fig, save_name)
    plt.show()

def makeConditionUncertaintyCoveragePlot(dirs, titles, variable_names, islice=None, jslice=None, figsize=(12,8), save_name=None):
    df = {"variable": [], "condition": [], "prediction standard deviation": [], "mean predictive uncertainty": [], "mean standard deviation coverage (%)": []}
    for j, d in enumerate(dirs):
        mean_path, variance_path = getPaths(d, returns=["mean","variance"])
        mean = loadRaster(mean_path, islice=islice, jslice=jslice)
        pu = loadRaster(variance_path, islice=islice, jslice=jslice, elementwise_fn=np.sqrt)
        pred_std = np.nanstd(mean, axis=(1,2))
        mean_pu = np.nanmean(pu, axis=(1,2))
        for i, variable_name in enumerate(variable_names):
            df["variable"].append(variable_name)
            df["condition"].append(titles[j])
            df["prediction standard deviation"].append(pred_std[i])
            df["mean predictive uncertainty"].append(mean_pu[i])
            df["mean standard deviation coverage (%)"].append(mean_pu[i]/pred_std[i]*100)
    df = pd.DataFrame(df)
    fig = plt.figure(figsize=figsize)
    sns.lineplot(
        data=df,
        x="condition",
        y="mean standard deviation coverage (%)",
        hue="variable"
    )
    plt.tight_layout()
    if save_name is not None: savefigure(fig, save_name)
    plt.show()
    
def showSinglePredictionMaps(dir_, title, variable_index, variable_name, s2repr_dirs, gt_dir, 
                                islice=None, jslice=None, normalize=False, save_name=None,
                                figsize=(15,2.5)):
    fig, axs = plt.subplots(nrows=1, ncols=6, figsize=figsize)
    d = dir_
    img_path, mean_path, variance_path, gt_path = getPaths(d, s2repr_dirs, gt_dir, returns=["img","mean","variance","gt"])
    # load rasters
    rgb = loadRaster(img_path, bands=[4,3,2], transpose_order=(1,2,0), clip_range=(100, 2000), islice=islice, jslice=jslice)
    gt = loadRaster(gt_path, bands=variable_index, islice=islice, jslice=jslice, set_nan_mask=True)
    if variable_index in [3,5]: gt /= 100 # Cover/Dens normalization!!
    mean = loadRaster(mean_path, bands=variable_index, islice=islice, jslice=jslice)
    predictive_std = loadRaster(variance_path, bands=variable_index, islice=islice, jslice=jslice, elementwise_fn=np.sqrt)
    if normalize:
        gt, mean, predictive_std = multiNorm2d(gt, mean, predictive_std)
        vmin, vmax = 0, 1
    else:
        vmin, vmax = multiMinMax(gt, mean, predictive_std)
    dvmin, dvmax = vmin-vmax, vmax-vmin
    rerror = mean-gt
    cerror = np.abs(rerror)-predictive_std
    rbound = np.nanmax(np.abs(rerror))
    cbound = np.nanmax(np.abs(cerror))
    axs[0].imshow(rgb)
    axs[0].set_title(f"rgb")
    sns.heatmap(gt, ax=axs[1], vmin=vmin, vmax=vmax)
    axs[1].set_title(f"gt")
    # break
    sns.heatmap(mean, ax=axs[2], vmin=vmin, vmax=vmax)
    axs[2].set_title(f"mean")
    sns.heatmap(predictive_std, ax=axs[3], vmin=np.nanmin(predictive_std), vmax=np.nanmax(predictive_std))
    axs[3].set_title(f"predictive\nuncertainty")
    sns.heatmap(rerror, ax=axs[4], cmap="bwr", vmin=-rbound, vmax=rbound)
    axs[4].set_title(f"r-error")
    sns.heatmap(cerror, ax=axs[5], cmap="bwr", vmin=-cbound, vmax=cbound)
    axs[5].set_title(f"c-error")
    for ax in axs.flatten(): ax.set_axis_off()
    fig.suptitle(f"{variable_name} - {title}")
    plt.tight_layout()
    if save_name is not None: savefigure(fig, save_name)
    plt.show()
    
def CompareMapsStats(matching_dirs, variable_names, split_mask=None, split=None):
    mean_stats = {"variable": [], "stat": [], "original": [], "gee": [], "delta": [], "relative delta (%)": []}
    pu_stats = {"variable": [], "stat": [], "original": [], "gee": [], "delta": [], "relative delta (%)": []}
    if split_mask is not None or split is not None: 
        assert split_mask is not None and split is not None
        split_ = split
        split = ["train", "val", "test"].index(split)
        mean_stats["split"] = []
        pu_stats["split"] = []    
    for i, (orig_dir, gee_dir) in enumerate(matching_dirs):
        # paths
        gee_mean_path, gee_variance_path = getPaths(gee_dir, returns=["mean", "variance"])
        orig_mean_path, orig_variance_path = getPaths(orig_dir, returns=["mean", "variance"])
        # means
        gee_mean = loadRaster(gee_mean_path)
        orig_mean = loadRaster(orig_mean_path)
        # variances
        gee_predictive_std = loadRaster(gee_variance_path, elementwise_fn=np.sqrt)
        orig_predictive_std = loadRaster(orig_variance_path, elementwise_fn=np.sqrt)
        n = gee_mean.shape[0]
        # select split
        if split_mask is not None: 
            gee_mean = gee_mean[:,split_mask==split]
            orig_mean = orig_mean[:,split_mask==split]
            gee_predictive_std = gee_predictive_std[:,split_mask==split]
            orig_predictive_std = orig_predictive_std[:,split_mask==split]
        else: 
            gee_mean = gee_mean.reshape(n, -1)
            orig_mean = orig_mean.reshape(n, -1)
            gee_predictive_std = gee_predictive_std.reshape(n, -1)
            orig_predictive_std = orig_predictive_std.reshape(n, -1)
        for f, fn in zip([np.nanmin, np.nanmax, np.nanmean, np.nanmedian, np.nanstd], 
                         ["min", "max", "mean", "median", "std"]):
            # mean
            mean_stats["variable"].extend(variable_names)
            mean_stats["stat"].extend([fn for _ in range(1, n+1)])
            mean_stats["original"].extend(f(orig_mean, axis=1).tolist())
            mean_stats["gee"].extend(f(gee_mean, axis=1).tolist())
            mean_stats["delta"].extend((f(gee_mean, axis=1)-f(orig_mean, axis=1)).tolist())
            mean_stats["relative delta (%)"].extend(
                ((f(gee_mean, axis=1)-f(orig_mean, axis=1))/f(orig_mean, axis=1)*100).tolist()
            )
            if split is not None:
                mean_stats["split"].extend([split_ for _ in range(1, n+1)])
            # variance
            pu_stats["variable"].extend(variable_names)
            pu_stats["stat"].extend([fn for _ in range(1, n+1)])
            pu_stats["original"].extend(f(orig_predictive_std, axis=1).tolist())
            pu_stats["gee"].extend(f(gee_predictive_std, axis=1).tolist())
            pu_stats["delta"].extend((f(gee_predictive_std, axis=1)-f(orig_predictive_std, axis=1)).tolist())
            pu_stats["relative delta (%)"].extend(
                ((f(gee_predictive_std, axis=1)-f(orig_predictive_std, axis=1))/f(orig_predictive_std, axis=1)*100).tolist()
            )
            if split is not None:
                pu_stats["split"].extend([split_ for _ in range(1, n+1)])
        return pd.DataFrame(mean_stats), pd.DataFrame(pu_stats)
    
def evaluateSplit(prediction_dir, gt_dir, split_mask, split):
    dir_name = prediction_dir.split("/")[-1]
    pid = dir_name.split("_")[0]
    split_id = ["train", "val", "test"].index(split)
    # Load data
    variance_path, mean_path, gt_path = getPaths(prediction_dir, gt_dir=gt_dir, returns=["variance", "mean", "gt"])
    variance = loadRaster(variance_path)
    mean = loadRaster(mean_path)
    gt = loadRaster(gt_path)
    variance[:,split_mask!=split_id] = np.nan
    mean[:,split_mask!=split_id] = np.nan
    gt[:,split_mask!=split_id] = np.nan
    gt[2] /= 100 # Cover/Dens normalization!!
    gt[4] /= 100
    rcu = StratifiedRCU(
        num_variables=variance.shape[0],
        num_groups=1,
        num_bins=1500,
        lo_variance=np.nanmin(variance, axis=(1,2)),
        hi_variance=np.nanmax(variance, axis=(1,2)),
    )
    rcu.add_project(pid, gt, mean, variance)
    return rcu
    
def loadMetricsDataFrame(matching_dirs, metrics=["mse", "ence", "auce", "cv"], 
                         variable_names=['P95', 'MeanH', 'Dens', 'Gini', 'Cover'],
                         split_mask=None, split=None, gt_dir=None):
    if split_mask is not None or split is not None: 
        assert split_mask is not None and split is not None and gt_dir is not None
    metric_query = " | ".join([f"metric == '{m}'" for m in metrics])
    # find matching rcus
    format_df = lambda df: (df.query("group != 'global' & kind == 'agg'")
                         .query(metric_query)
                         .drop(["group", "kind"], axis=1))
    data = {"metric": [],
            "variable": [],
            "imageId": [],
            "source": [],
            "x": []}
    if split: data["split"] = []
    for orig_dir, gee_dir in matching_dirs:
        # load dataframes
        if split_mask is not None:
            print(f"Re-evaluating GEE for {split}")
            gee_rcu = evaluateSplit(gee_dir, gt_dir, split_mask, split)
            print(f"Re-evaluating original for {split}")
            orig_rcu = evaluateSplit(orig_dir, gt_dir, split_mask, split)
        else:
            gee_rcu = StratifiedRCU.from_json(os.path.join(gee_dir.path, "rcu.json"))
            orig_rcu = StratifiedRCU.from_json(os.path.join(orig_dir.path, "rcu.json"))
        gee_rcu.get_results_df(groups={}, variable_names=variable_names)
        orig_rcu.get_results_df(groups={}, variable_names=variable_names)
        gee_df = format_df(gee_rcu.results)
        orig_df = format_df(orig_rcu.results)
        # get results
        for gt, ot in zip(gee_df.itertuples(), orig_df.itertuples()):
            assert gt.metric==ot.metric
            assert gt.variable==ot.variable
            data["metric"].extend([gt.metric, ot.metric])
            data["variable"].extend([gt.variable, ot.variable])
            data["imageId"].extend([Path(gee_dir).name, Path(orig_dir).name])
            data["source"].extend(["gee", "original"])
            data["x"].extend([gt.x, ot.x])
            if split: data["split"].extend([split, split])
    fulldf = pd.DataFrame(data)
    return fulldf

def plotTrainTestMetricsDataFrames(train_df, test_df, type="bar", save_name=None, 
                                   variables=None, split_splits=True, figsize=(15,10)):
    assert type in ["bar", "scatter"]
    metrics_df = pd.concat([train_df, test_df])
    variables = variables if variables is not None else metrics_df.variable.unique()
    fig, axs = plt.subplots(
        nrows=len(metrics_df.metric.unique()),
        ncols=len(variables),
        figsize=figsize
    )
    for i, m in enumerate(metrics_df.metric.unique()):
        for j, v in enumerate(variables):
            tmp = (metrics_df
                   .query(f"metric == '{m}' & variable == '{v}'")
                   .drop(columns=["metric", "variable", "imageId"]))
            mtmp = (tmp
               .groupby(by=["source", "split"])
               .mean()
               .reset_index())
            xx = "split" if split_splits else None
            hue = "source" if split_splits else "split"
            if type == "scatter":
                sns.scatterplot(data=tmp, x=xx, y="x", hue=hue, ax=axs[i,j], marker="o", s=20, alpha=0.4)
                sns.scatterplot(data=mtmp, x=xx, y="x", hue=hue, ax=axs[i,j], marker="_", s=100, linewidth=2)
                axs[i,j].set(xlim=(-0.5, 1.5))
            else:
                kw = {"hue_order": ["original", "gee"]} if split_splits else {}
                sns.barplot(data=tmp, x=xx, y="x", hue=hue, ax=axs[i,j], 
                            errorbar=None, **kw)
            try: axs[i,j].get_legend().remove()
            except: continue
            axs[i,j].set_xlabel("")
            axs[i,j].set_ylabel("")
            if i==0:
                axs[i,j].set_title(v)
            if j==0:
                axs[i,j].set_ylabel(m)
    plt.tight_layout()
    fig = single_legend_figure(fig, 2)
    if save_name is not None: savefigure(fig, save_name)
    plt.show()

def showCloudVsUncertaintyType(
    dirs,
    s2repr_dirs,
    variable_names,
    figsize=(12,18),
    save_name=None
):
    nv = len(variable_names)
    df = pd.DataFrame(
        columns=["cp", "count", "average uncertainty", "uncertainty std", "type", "variable"])
    # loop on directories
    for i, d in enumerate(dirs):
        # read data
        aleatoric_path, epistemic_path, img_path = getPaths(
            d, s2repr_dirs=s2repr_dirs, returns=["aleatoric", "epistemic", "img"]
        )
        aleatoric = loadRaster(aleatoric_path, dtype="float64", elementwise_fn=np.sqrt).reshape(nv, -1)
        epistemic = loadRaster(epistemic_path, dtype="float64", elementwise_fn=np.sqrt).reshape(nv, -1)
        cp = loadRaster(img_path, bands=-1, dtype="float").reshape(-1)
        mask = np.bitwise_and(~np.isnan(aleatoric).all(0), ~np.isnan(epistemic).all(0))
        aleatoric, epistemic, cp = aleatoric[:,mask], epistemic[:,mask], cp[mask]
        tmp = {
            "cp": [],
            "count": [],
            "average uncertainty": [],
            "uncertainty std": [],
            "type": [],
            "variable": [],
        }
        for i, variable_name in enumerate(variable_names):
            for array, array_type in zip([aleatoric[i], epistemic[i]], ["aleatoric", "epistemic"]):
                tmp["cp"].extend(cp.tolist())
                tmp["count"].extend(np.ones_like(cp).tolist())
                tmp[f"average uncertainty"].extend(array.tolist())
                tmp[f"uncertainty std"].extend((array**2).tolist())
                tmp[f"type"].extend([array_type for _ in array])
                tmp[f"variable"].extend([variable_name for _ in array])
        tmp = pd.DataFrame(tmp)
        tmp = tmp.groupby(["cp", "type", "variable"]).sum().reset_index()
        df = pd.concat([df, tmp], axis=0)
    df = df.groupby(["cp", "variable", "type"]).sum().reset_index()
    df["average uncertainty"] /= df["count"]
    df["uncertainty std"] /= df["count"]
    df["uncertainty std"] = np.sqrt(df["uncertainty std"]-df["average uncertainty"]**2)
    fig, axs = plt.subplots(ncols=3, nrows=nv, figsize=figsize)
    for i, variable_name in enumerate(variable_names):
        sub = df.query(f"variable == '{variable_name}'")
        sns.lineplot(data=sub, x="cp", y="average uncertainty", hue="type", ax=axs[i,0])
        axs[i,0].fill_between(sub[sub.type=="aleatoric"].cp, 
                        sub[sub.type=="aleatoric"]["average uncertainty"]-sub[sub.type=="aleatoric"]["uncertainty std"], 
                        sub[sub.type=="aleatoric"]["average uncertainty"]+sub[sub.type=="aleatoric"]["uncertainty std"], 
                        color="b",
                        alpha=0.2)
        axs[i,0].fill_between(sub[sub.type=="epistemic"].cp, 
                        sub[sub.type=="epistemic"]["average uncertainty"]-sub[sub.type=="epistemic"]["uncertainty std"], 
                        sub[sub.type=="epistemic"]["average uncertainty"]+sub[sub.type=="epistemic"]["uncertainty std"], 
                        color="orange",
                        alpha=0.2)
        axs[i,0].set_ylabel(f"{variable_name}")
        axs[i,0].set_xlabel(f"cloud probability")
        if i==0: axs[i,0].set_title(f"average uncertainty")
        if i==nv-1: axs[i,0].legend().set_title("")
        else: axs[i,0].legend([],[],frameon=False)
    for i, variable_name in enumerate(variable_names):
        for utype in ["aleatoric", "epistemic"]:
            sub = df.query(f"variable == '{variable_name}' & type == '{utype}'")
            sub["uncertainty change"] = sub["average uncertainty"].values-sub["average uncertainty"].values[0]
            c = "tab:orange" if utype=="epistemic" else "tab:blue"
            axs[i,1].plot(sub.cp, sub["uncertainty change"], color=c, label=utype)
        if i==nv-1: axs[i,1].legend()
        axs[i,1].set_xlabel(f"cloud probability")
        if i==0: axs[i,1].set_title(f"average uncertainty increase")
    df["uncertainty_fraction"] = [None for _ in range(df.shape[0])]
    total_avg_uncertainty = df.groupby(["variable", "cp"]).sum()["average uncertainty"].reset_index()
    for i, variable_name in enumerate(variable_names):
        for utype in ["aleatoric", "epistemic"]:
            values = df.query(f"variable == '{variable_name}' & type == '{utype}'").sort_values("cp")
            fracs = values["average uncertainty"].values / total_avg_uncertainty.query(f"variable == '{variable_name}'").sort_values("cp")["average uncertainty"].values
            c = "tab:orange" if utype=="epistemic" else "tab:blue"
            axs[i,2].plot(values.cp, fracs, color=c, label=utype)
        axs[i,2].hlines(0.5, xmin=0, xmax=100, linestyle="dotted", color="black")
        if i==nv-1: axs[i,2].legend()
        axs[i,2].set_xlabel(f"cloud probability")
        if i==0: axs[i,2].set_title(f"average uncertainty fraction")
    plt.tight_layout()
    if save_name is not None: savefigure(fig, save_name)
    plt.show()

def showCloudVsPrediction(
    dirs,
    s2repr_dirs,
    variable_names,
    trainset_means,
    islice=None,
    jslice=None,
    figsize=(12,18),
    save_name=None
):
    nv = len(variable_names)
    df = pd.DataFrame(
        columns=["cp", "count", "average", "stddev", "variable"])
    # loop on directories
    for i, d in enumerate(dirs):
        # read data
        mean_path, img_path = getPaths(
            d, s2repr_dirs=s2repr_dirs, returns=["mean", "img"]
        )
        mean = loadRaster(mean_path, dtype="float64", islice=islice, jslice=jslice).reshape(nv, -1)
        cp = loadRaster(img_path, bands=-1, islice=islice, jslice=jslice, dtype="float").reshape(-1)
        mask = ~np.isnan(mean).all(0)
        mean, cp = mean[:,mask], cp[mask]
        tmp = {
            "cp": [],
            "count": [],
            "average": [],
            "stddev": [],
            "variable": [],
        }
        for i, variable_name in enumerate(variable_names):
            tmp["cp"].extend(cp.tolist())
            tmp["count"].extend(np.ones_like(cp).tolist())
            tmp[f"average"].extend(mean[i].tolist())
            tmp[f"stddev"].extend((mean[i]**2).tolist())
            tmp[f"variable"].extend([variable_name for _ in mean[i]])
        tmp = pd.DataFrame(tmp)
        tmp = tmp.groupby(["cp", "variable"]).sum().reset_index()
        df = pd.concat([df, tmp], axis=0)
    df = df.groupby(["cp", "variable"]).sum().reset_index()
    df["average"] /= df["count"]
    df["stddev"] /= df["count"]
    df["stddev"] = np.sqrt(df["stddev"]-df["average"]**2)
    fig, axs = plt.subplots(ncols=2, nrows=3, figsize=figsize)
    axs = axs.flatten()
    for i, variable_name in enumerate(variable_names):
        sub = df.query(f"variable == '{variable_name}'")
        sns.lineplot(data=sub, x="cp", y="average", ax=axs[i])
        axs[i].fill_between(sub.cp, sub.average-sub.stddev, sub.average+sub.stddev, alpha=0.2) 
        axs[i].hlines(trainset_means[i], xmin=0, xmax=100, linestyle="dotted", color="black", label=f"train set mean ({trainset_means[i]:.3f})")
        axs[i].set_ylabel("average prediction")
        axs[i].set_xlabel(f"cloud probability")
        axs[i].set_title(variable_name)
        axs[i].legend()
    fig.delaxes(axs.flatten()[-1])
    plt.tight_layout()
    if save_name is not None: savefigure(fig, save_name)
    plt.show()

def showCloudVsUncertaintyCoverage(
    dirs,
    s2repr_dirs,
    variable_names,
    trainset_stds,
    islice=None,
    jslice=None,
    figsize=(12,8),
    save_name=None
):
    nv = len(variable_names)
    df = pd.DataFrame(
        columns=["cp", "count", "average", "stddev", "variable"])
    # loop on directories
    for i, d in enumerate(dirs):
        # read data
        pu_path, img_path = getPaths(
            d, s2repr_dirs=s2repr_dirs, returns=["variance", "img"]
        )
        pu = loadRaster(pu_path, dtype="float64", islice=islice, jslice=jslice, elementwise_fn=np.sqrt).reshape(nv, -1)
        cp = loadRaster(img_path, bands=-1, islice=islice, jslice=jslice, dtype="float").reshape(-1)
        mask = ~np.isnan(pu).all(0)
        pu, cp = pu[:,mask], cp[mask]
        tmp = {
            "cp": [],
            "count": [],
            "average": [],
            "stddev": [],
            "variable": [],
        }
        for i, variable_name in enumerate(variable_names):
            tmp["cp"].extend(cp.tolist())
            tmp["count"].extend(np.ones_like(cp).tolist())
            tmp[f"average"].extend(pu[i].tolist())
            tmp[f"stddev"].extend((pu[i]**2).tolist())
            tmp[f"variable"].extend([variable_name for _ in pu[i]])
        tmp = pd.DataFrame(tmp)
        tmp = tmp.groupby(["cp", "variable"]).sum().reset_index()
        df = pd.concat([df, tmp], axis=0)
    df = df.groupby(["cp", "variable"]).sum().reset_index()
    df.loc[:,"average"] /= df["count"]
    df.loc[:,"stddev"] /= df["count"]
    df["stddev"] = np.sqrt(df["stddev"]-df["average"]**2)
    fig = plt.figure(figsize=figsize)
    ax = plt.gca()
    colors = sns.color_palette()
    for i, variable_name in enumerate(variable_names):
        sub = df.query(f"variable == '{variable_name}'")
        sub.loc[:,"average"] /= trainset_stds[i]
        sub.loc[:,"stddev"] /= trainset_stds[i]
        sub.loc[:,"average"] *= 100
        sub.loc[:,"stddev"] *= 100
        sns.lineplot(data=sub, x="cp", y="average", ax=ax, color=colors[i])
        ax.fill_between(sub.cp, sub.average-sub.stddev, sub.average+sub.stddev, alpha=0.1, color=colors[i])
        ax.set_ylabel("mean standard deviation coverage (%)")
        ax.set_xlabel(f"cloud probability")
        ax.set_title(variable_name)
    plt.tight_layout()
    if save_name is not None: savefigure(fig, save_name)
    plt.show()
    return df

def getUsabilityData(
    dirs,
    s2repr_dirs,
    gt_dir,
    num_variables,
    variance_bounds=None, # used to discard distribution tail
    islice=None,
    jslice=None
):
    """
    returns
        variances
        cloud_probabilities
        rerrors
        cerrors
    """
    cps, variances, rerrors, cerrors = tuple([[] for _ in range(num_variables)] for _ in range(4))
    for dir_ in dirs:
        # paths
        mean_path, variance_path, gt_path, img_path = getPaths(dir_, s2repr_dirs, gt_dir, returns=["mean","variance","gt", "img"])
        # load rasters
        gt = loadRaster(gt_path, islice=islice, jslice=jslice, set_nan_mask=True)
        gt[2] /= 100
        gt[4] /= 100
        mean = loadRaster(mean_path, islice=islice, jslice=jslice)
        variance = loadRaster(variance_path, islice=islice, jslice=jslice)
        cp = loadRaster(img_path, bands=-1, islice=islice, jslice=jslice)
        # compute valid mask
        cps_list, variances_list, rerrors_list, cerrors_list = [], [], [], []
        for i in range(mean.shape[0]):
            # compute valid mask
            valid = ~np.isnan(cp)
            variancei = variance[i]
            if variance_bounds is not None:
                variance_bounds[i] = [float(vb) for vb in variance_bounds[i]]
                variancei[variancei<variance_bounds[i][0]] = np.nan
                variancei[variancei>variance_bounds[i][1]] = np.nan
            for x in [gt[i], mean[i], variance[i]]: 
                valid = np.bitwise_and(valid, ~np.isnan(x))
            # mask by variable
            cpi = cp[valid]
            gti = gt[i,valid]
            meani = mean[i,valid]
            variancei = variancei[valid]
            gti = gt[i,valid]
            rerrori = np.abs(meani-gti)
            cerrori = np.abs(np.sqrt(rerrori**2)-np.sqrt(variancei))
            # add to acc
            cps[i].extend(cpi)
            variances[i].extend(variancei)
            rerrors[i].extend(rerrori)
            cerrors[i].extend(cerrori)
    cps = [np.array(x) for x in cps]
    variances = [np.array(x) for x in variances]
    rerrors = [np.array(x) for x in rerrors]
    cerrors = [np.array(x) for x in cerrors]
    return cps, variances, rerrors, cerrors

def showMetricsCloudRepartition(
    cps,
    rerrors,
    cerrors,
    variable_names,
    cloud_probability_bins=[0, 1, 20, 99],
    figsize=(20,17),
    save_name=None
):
    bins = cloud_probability_bins
    fig, axs = plt.subplots(nrows=len(variable_names), ncols=2, figsize=figsize)
    for i, variable_name in enumerate(variable_names):
        cp = cps[i]
        rerror = rerrors[i]
        cerror = cerrors[i]
        str_bins = [f"[{bins[i]}, {bins[i+1]})" for i in range(len(bins)-1)]+[f"[{bins[-1]}, 100]"]
        # compute bin assignments
        bin_ids = np.digitize(cp, bins=np.array(bins))-1
        binned_cp = np.array([str_bins[i] for i in bin_ids])
        # count bin
        bin_counts = [(bin_ids==bin_id).sum() for bin_id in np.unique(bin_ids)]
        # sample bins of equal size
        sample_size = min(bin_counts)
        # dataframe
        df = pd.DataFrame({
            "rerror": rerror,
            "cerror": cerror,
            "cloud probability": binned_cp,
        })
        # plot
        sns.histplot(
            data=df, 
            x="rerror",
            hue="cloud probability",
            multiple="fill",
            bins=8,
            stat="density",
            hue_order=str_bins,
            common_norm=True,
            ax=axs[i,0]
        )
        axs[i,0].set_xlabel(f"{variable_name} regression error")
        axs[i,0].set_ylabel("Density")
        sns.histplot(
            data=df, 
            x="cerror",
            hue="cloud probability",
            multiple="fill",
            bins=8,
            stat="density",
            hue_order=str_bins,
            common_norm=True,
            ax=axs[i,1]
        )
        axs[i,1].set_xlabel(f"{variable_name} calibration error")
        axs[i,1].set_ylabel("")
    fig.tight_layout()
    if save_name is not None: savefigure(fig, save_name)
    plt.show()
    
def showPredictiveUncertaintyCloudRepartition(
    cps,
    variances,
    variable_names,
    cloud_probability_bins=[0, 1, 20, 99],
    figsize=(12,17),
    save_name=None
):
    bins = cloud_probability_bins
    fig, axs = plt.subplots(nrows=len(variable_names), ncols=1, figsize=figsize)
    for i, variable_name in enumerate(variable_names):
        cp = cps[i]
        predictive_std = np.sqrt(variances[i])
        str_bins = [f"[{bins[i]}, {bins[i+1]})" for i in range(len(bins)-1)]+[f"[{bins[-1]}, 100]"]
        # compute bin assignments
        bin_ids = np.digitize(cp, bins=np.array(bins))-1
        binned_cp = np.array([str_bins[i] for i in bin_ids])
        # count bin
        bin_counts = [(bin_ids==bin_id).sum() for bin_id in np.unique(bin_ids)]
        # create equal bins
        sample_size = min(bin_counts)
        # dataframe
        df = pd.DataFrame({
            "pu": predictive_std,
            "cloud probability": binned_cp,
        })
        # plot
        sns.histplot(
            data=df, 
            x="pu",
            hue="cloud probability",
            multiple="fill",
            bins=8,
            stat="density",
            hue_order=str_bins,
            common_norm=True,
            ax=axs[i]
        )
        axs[i].set_xlabel(f"{variable_name} predictive uncertainty")
        axs[i].set_ylabel("Density")
    fig.tight_layout()
    if save_name is not None: savefigure(fig, save_name)
    plt.show()

def getCloudlessVisualizer(
    dirs,
    gt_dir, 
    s2repr_dirs,
    cloudy_pixel_threshold,
    cloudy_exp_vars,
    exp_var_name,
    variable_names
):
    rcus = []
    for dir_ in dirs:
        dir_name = dir_.split("/")[-1]
        pid = dir_name.split("_")[0]
        # paths
        img_path, mean_path, variance_path, gt_path = getPaths(dir_, s2repr_dirs, gt_dir, returns=["img","mean","variance","gt"])
        # read data
        gt = loadRaster(gt_path, dtype="float")
        gt[2] /= 100
        gt[4] /= 100
        cp = loadRaster(img_path, bands=-1, dtype="float")
        mean = loadRaster(mean_path, dtype="float")
        variance = loadRaster(variance_path, dtype="float")
        # Mask out clouds according to threshold
        cmask = cp>cloudy_pixel_threshold
        gt[:,cmask] = np.nan
        mean[:,cmask] = np.nan
        variance[:,cmask] = np.nan
        # create single img rcu
        rcu = StratifiedRCU(
            num_variables=variance.shape[0],
            num_groups=1,
            num_bins=1500,
            lo_variance=np.nanmin(variance, axis=(1,2)),
            hi_variance=np.nanmax(variance, axis=(1,2)),
        )
        rcu.add_project(pid, gt, mean, variance)
        rcu.get_results_df(groups={}, variable_names=variable_names)
        rcus.append(rcu)
    visualizer = ExperimentVisualizer(
        rcus=rcus,
        exp_var_name=exp_var_name,
        exp_vars=cloudy_exp_vars,
        variable_names=variable_names,
        fig_ncols=2
    )
    return visualizer

def showCloudyCloudlessMetrics(
    cloudy_visualizer, 
    cloudless_visualizer,
    bins, 
    metrics, 
    kind="agg",
    figsize=(12,18),
    ncols=2,
    save_name=None
):
    assert cloudy_visualizer.exp_var_name==cloudless_visualizer.exp_var_name
    assert cloudy_visualizer.exp_vars==cloudless_visualizer.exp_vars
    assert cloudy_visualizer.variable_names==cloudless_visualizer.variable_names
    bin_strings = [f"{bins[i]}-{bins[i+1]}" for i in range(len(bins)-1)]
    num_variables = len(cloudy_visualizer.variable_names)
    nrows = 1 if ncols>=num_variables else math.ceil(num_variables/ncols)
    for metric in metrics:
        # create figure
        fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
        axs = axs.flatten()
        # variables loop
        for i, var in enumerate(cloudy_visualizer.variable_names):
            ## REPLACE: self.variable_metric_boxplot(metric, var, kind, exp_var_bins, axs[i], fig_ncols)
            query = " & ".join([
                f"kind == '{kind.strip()}'",
                f"metric == '{metric.strip()}'",
                f"variable == '{var.strip()}'",
                "group == 'global'"
            ])
            cloudy_df = (cloudy_visualizer.df.copy()
                         .query(query)
                         .drop(["kind", "metric", "variable", "group"], axis=1)
                         .assign(group=lambda x: "cloudy"))
            cloudless_df = (cloudless_visualizer.df.copy()
                         .query(query)
                         .drop(["kind", "metric", "variable", "group"], axis=1)
                         .assign(group=lambda x: "cloudless"))
            df = pd.concat([cloudy_df, cloudless_df])
            df = df[df[cloudy_visualizer.exp_var_name]<=np.max(bins)]
            bin_ids = np.digitize(
                df[cloudy_visualizer.exp_var_name].values, 
                bins=bins
            )
            bin_ids = bin_ids[bin_ids-1<len(bin_strings)]
            df.loc[:,cloudy_visualizer.exp_var_name] = [
                bin_strings[i-1] 
                for i in bin_ids
            ]
            sns.boxplot(
                data=df, y="x", x=cloudy_visualizer.exp_var_name, 
                ax=axs[i], hue="group", 
                order=bin_strings, 
                hue_order=["cloudless", "cloudy"],
                showfliers=False
            )
            ## 
            axs[i].set_title(var)
            axs[i].set_ylabel("")
        fig.suptitle(metric)
        if num_variables%ncols!=0: fig.delaxes(axs.flatten()[-1])
        if save_name is not None: 
            fig.tight_layout()
            savefigure(fig, save_name+f"_{metric}")
        plt.show()

def cloudyCloudlessMetrics(
    cloudy_visualizer, 
    cloudless_visualizer,
    bins, 
    metrics, 
    kind="agg",
    save_name_suffix=None
):
    # default bins: np.array([0, 2, 15, 50, 100])
    if save_name_suffix is not None:
        save_name = f"images/cloud_experiment/cloudy_cloudless_{save_name_suffix}"
    else: save_name = f"images/cloud_experiment/cloudy_cloudless"
    showCloudyCloudlessMetrics(
        cloudy_visualizer, 
        cloudless_visualizer,
        bins=bins, 
        metrics=metrics, 
        kind=kind,
        figsize=(12,18),
        ncols=2,
        save_name=save_name
    )

# def histogram2dplot(
#     xarray_generator,
#     yarray_generator,
#     xbins,
#     ybins,
#     log_counts=False,
#     **hm_kwargs
# ):
#     H = None
#     xbounds = (np.inf, -np.inf)
#     ybounds = (np.inf, -np.inf)
#     # Compute histogram from generator
#     for xarray, yarray in zip(xarray_generator, yarray_generator):
#         # remove nan
#         valid = np.bitwise_and(~np.isnan(xarray), ~np.isnan(yarray))
#         xarray, yarray = xarray[valid].astype(int), yarray[valid].astype(int)
#         # compute histogram
#         h, xbnds, ybnds = np.histogram2d(x=xarray, y=yarray, bins=[xbins, ybins])
#         h = h[::-1,:]
#         if H is None: H = h
#         else: H += h
#         xbounds = min(xbounds[0], xbnds[0]), max(xbounds[1], xbnds[-1])
#         ybounds = min(ybounds[0], ybnds[0]), max(ybounds[1], ybnds[-1])
#     # ticks
#     xticks = [0, len(xbins)-1]
#     yticks = [0, len(ybins)-1]
#     print(xticks, yticks)
#     xticks_labels = [int(xbounds[0]), int(xbounds[1]-1)]
#     yticks_labels = [int(ybounds[1]-1), int(ybounds[0])]
#     if log_counts: 
#         # hm_kwargs["norm"] = LogNorm
#         mx = H.max()
#         H = np.log(H+1)
#         cbar_kws={"ticks": [0, mx], "format": "%.e"}
#     else: cbar_kws = {}
#     ax = hm_kwargs.get("ax", plt.gca())
#     sns.heatmap(H, cbar_kws=cbar_kws, **hm_kwargs)
#     ax.set_xticks(xticks, [math.floor(xbins.min()), xbins.max()])
#     ax.set_yticks(yticks, [ybins.max(), math.floor(ybins.min())])
#     return ax

# def varianceGenerator(dirs, index, bounds):
#     for dir_ in dirs:
#         dir_name = dir_.split("/")[-1]
#         pid = dir_name.split("_")[0]
#         variance_path = Path(os.path.join(dir_, f"{pid}_variance.tif"))
#         f = rasterio.open(variance_path)
#         v = f.read(index).astype("float")
#         f.close()
#         v[v<bounds[0]] = np.nan
#         v[v>bounds[1]] = np.nan
#         yield v

# def cpGenerator(dirs, s2repr_dirs, bounds):
#     for dir_ in dirs:
#         dir_name = dir_.split("/")[-1]
#         pid = dir_name.split("_")[0]
#         img_path = list(Path(os.path.join(s2repr_dirs, dir_name, pid)).glob("*.tif"))[0]
#         f = rasterio.open(img_path)
#         cp = f.read(f.count).astype("float")
#         f.close()
#         cp[cp<bounds[0]] = np.nan
#         cp[cp>bounds[1]] = np.nan
#         yield cp

# def showVarianceCloudMap(
#     dirs,
#     s2repr_dirs,
#     variable_index,
#     variable_name,
#     nbins_variance,
#     nbins_cloud_probability,
#     variance_bounds=None,
#     cloud_probability_bounds=[0,100],
#     save_name=None,
#     figsize=(8,8),
#     log_counts=False,
#     magnification=None,
# ):
#     # Compute variance bounds if required
#     if variance_bounds is None:
#         hi, lo = -np.inf, np.inf
#         for dir_ in dirs:
#             dir_name = dir_.split("/")[-1]
#             pid = dir_name.split("_")[0]
#             variance_path = Path(os.path.join(dir_, f"{pid}_variance.tif"))
#             img_path = list(Path(os.path.join(s2repr_dirs, dir_name, pid)).glob("*.tif"))[0]
#             with rasterio.open(img_path) as f: valid = ~np.isnan(f.read(f.count))
#             with rasterio.open(variance_path) as f: variance = f.read(variable_index).astype("float")
#             valid = np.bitwise_and(valid, ~np.isnan(variance))
#             variance = variance[valid]
#             hi = max(np.nanmax(variance), hi)
#             lo = min(np.nanmin(variance), lo)
#         variance_bounds = [lo, hi]
#     assert len(variance_bounds)==2, "must provide variance bounds as min/max"
#     # CP bounds
#     assert len(cloud_probability_bounds)==2, "must provide cloud probability bounds as min/max"
#     # Compute hisotgram on the fly
#     H = None
#     varbins = np.linspace(variance_bounds[0], variance_bounds[1], nbins_variance+1)
#     cpbins = np.linspace(cloud_probability_bounds[0], cloud_probability_bounds[1], nbins_cloud_probability+1)
#     for dir_ in dirs:
#         # read nonan
#         dir_name = dir_.split("/")[-1]
#         pid = dir_name.split("_")[0]
#         variance_path = Path(os.path.join(dir_, f"{pid}_variance.tif"))
#         img_path = list(Path(os.path.join(s2repr_dirs, dir_name, pid)).glob("*.tif"))[0]
#         with rasterio.open(img_path) as f: 
#             cp = f.read(f.count)
#         with rasterio.open(variance_path) as f: 
#             variance = f.read(variable_index)
#         valid = np.bitwise_and(cp>=cloud_probability_bounds[0], cp<=cloud_probability_bounds[1])
#         valid = np.bitwise_and(valid, variance>=variance_bounds[0])
#         valid = np.bitwise_and(valid, variance<=variance_bounds[1])
#         valid = np.bitwise_and(valid, ~np.isnan(cp))
#         valid = np.bitwise_and(valid, ~np.isnan(variance))
#         variance = variance[valid]
#         cp = cp[valid]
#         # local histogram
#         h = np.histogram2d(x=cp, y=variance, bins=[cpbins, varbins])[0]
#         h = h[::-1,:]
#         if H is None: H = h
#         else: H += h
#     # if log_counts: H = np.log(H+1)
#     # plot
#     if magnification is not None:
#         from scipy.interpolate import interp2d
#         f = interp2d(
#             np.linspace(*cloud_probability_bounds, nbins_cloud_probability), # 0, bound, nbins_cp
#             np.linspace(*variance_bounds, nbins_variance), # 0, bound, nbins_var
#             H, kind="linear"
#         )
#         nbins_cloud_probability *= magnification
#         nbins_variance *= magnification
#         H = f(
#             np.linspace(*cloud_probability_bounds, nbins_cloud_probability), # 0, bound, nbins_cp
#             np.linspace(*variance_bounds, nbins_variance), # 0, bound, nbins_var
#         )
#     fig = plt.figure(figsize=figsize)
#     ax = plt.gca()
#     if log_counts: 
#         H = np.log(H+1)
#         sns.heatmap(H, ax=ax, cbar=True)
#     else: sns.heatmap(H, ax=ax, cbar=True)
#     ax.set_xticks([0, nbins_variance], [math.floor(variance_bounds[0]), math.ceil(variance_bounds[1])], rotation=0)
#     ax.set_yticks([0, nbins_cloud_probability], [cloud_probability_bounds[1],cloud_probability_bounds[0]])
#     ax.set_title(variable_name)
#     ax.set_xlabel("variance")
#     ax.set_ylabel("cloud probability")
#     # if log_counts: 
#     #     import matplotlib.cm as cm
#     #     H[H==0] = 1e-2
#     #     H = np.log(H)
#     #     pcm = cm.ScalarMappable(norm=LogNorm(H.min(), H.max()))
#     #     fig.colorbar(pcm, ax=ax)
#     if save_name is not None: savefigure(fig, save_name)
#     plt.tight_layout()
#     plt.show()

