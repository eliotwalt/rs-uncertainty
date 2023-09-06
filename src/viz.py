import wandb
import pandas as pd
import matplotlib.pyplot as plt 
import matplotlib.patches as patches
from matplotlib.ticker import MaxNLocator
from matplotlib.colors import LogNorm
import tempfile
import math
import json
from scipy.signal import decimate
import scipy.stats
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
    
    def variable_histogram_plot(self, variable, ax=None, hi_bound=np.inf, log=True, palette=None, show_legend=True, show_cbar=True):
        if ax is None:
            fig, ax = plt.subplots()
        var_idx = self.variable_names.index(variable)
        hdf = pd.DataFrame(columns=[self.exp_var_name, "probability", "predictive uncertainty"])
        for i, rcu in enumerate(self.rcus):
            H = np.nansum(rcu.histogram[0,var_idx], axis=0)
            ct = np.full(H.shape, str(self.exp_vars[i]))
            bins = np.sqrt(rcu.histogram.bins[var_idx])
            hdf = pd.concat([hdf, pd.DataFrame({self.exp_var_name: ct, "probability": H/np.nansum(H), "predictive uncertainty": bins})])
            hdf = hdf[hdf["predictive uncertainty"]<hi_bound]
        hhdf = hdf.groupby(["predictive uncertainty", self.exp_var_name]).sum().reset_index()
        hhdf = hhdf.sort_values(by=[self.exp_var_name])
        if palette is not None: kwargs = {"palette": palette}
        else: kwargs = {}
        g = sns.lineplot(data=hhdf, x="predictive uncertainty", hue=self.exp_var_name, y="probability", ax=ax, alpha=0.5, **kwargs)
        if log: 
            ax.set_xscale("log")
            ax.set_xlabel("log predictive uncertainty")
        if not show_legend:
            ax.get_legend().remove()
            cmap = sns.color_palette("bwr")
            b, r = cmap[0], cmap[-1]
            argmax_expvar = np.argmax(self.exp_vars)
            max_rcu = self.rcus[argmax_expvar]
            prob = np.nansum(max_rcu.histogram[0,var_idx], axis=0)/np.nansum(max_rcu.histogram[0,var_idx])
            bins = np.sqrt(max_rcu.histogram.bins[var_idx])
            mask = bins<hi_bound
            bins, prob = bins[mask], prob[mask]
            ax.plot(bins, prob, color=r, label=f"maximum: {self.exp_vars[argmax_expvar]:.2f}%", alpha=1)
            argmin_expvar = np.argmin(self.exp_vars)
            min_rcu = self.rcus[argmin_expvar]
            prob = np.nansum(min_rcu.histogram[0,var_idx], axis=0)/np.nansum(min_rcu.histogram[0,var_idx])
            bins = np.sqrt(min_rcu.histogram.bins[var_idx])
            mask = bins<hi_bound
            bins, prob = bins[mask], prob[mask]
            ax.plot(bins, prob, color=b, label=f"minimum: {self.exp_vars[argmin_expvar]:.2f}%", alpha=1)
            handles, labels = ax.get_legend_handles_labels()
            ax.legend([handle for i,handle in enumerate(handles[-2:])],
                    [label for i,label in enumerate(labels[-2:])],
                    title="average cloud probability",
                    loc= 'best')
        else:
            ax.legend(loc="upper right")
        if show_cbar:
            sm = plt.cm.ScalarMappable(cmap="bwr", norm=plt.Normalize(min(self.exp_vars), max(self.exp_vars)))
            ax.figure.colorbar(sm, ax=ax)
        return ax
    
    def histogram_plot(self, hi_bounds=np.inf, log=True, figsize=(12, 18), save_name=None, fig_ncols=None, variables=None, add_title=True, **kwargs):
        if variables is None: variables = self.variable_names
        if not isinstance(hi_bounds, list): hi_bounds = [hi_bounds for _ in range(len(variables))]
        num_variables = len(variables)
        if fig_ncols is not None:
            previous_ncols = self.fig_ncols
            self.fig_ncols = fig_ncols
        ncols = self.fig_ncols
        nrows = 1 if self.fig_ncols>=num_variables else math.ceil(num_variables/self.fig_ncols)
        fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
        if len(variables)>1: axs = axs.flatten()
        for i, var in enumerate(variables):
            ax = axs[i] if len(variables)>1 else axs
            self.variable_histogram_plot(var, ax, hi_bounds[i], log, show_cbar=i==ncols-1, **kwargs)
            if add_title: ax.set_title(var)
        if num_variables%self.fig_ncols!=0: fig.delaxes(axs.flatten()[-1])
        if fig_ncols is not None: self.fig_ncols = previous_ncols
        fig.tight_layout()
        if save_name is not None: savefigure(fig, save_name)
        plt.show()

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
        fig.tight_layout()
        if save_name is not None: savefigure(fig, save_name)
        plt.show()

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
            sns.boxplot(data=gvmdf, y="x", x=self.exp_var_name, ax=ax, order=bin_strings, showfliers=False, color=sns.color_palette()[0])
        ax.set_ylabel(metric)
        return ax

    def metric_boxplot(self, metric, kind, exp_var_bins=None, figsize=(12, 18), fig_ncols=None, save_name=None, add_title=True):
        num_variables = len(self.variable_names)
        if fig_ncols is not None:
            previous_ncols = self.fig_ncols
            self.fig_ncols = fig_ncols
        ncols = self.fig_ncols
        nrows = 1 if self.fig_ncols>=num_variables else math.ceil(num_variables/self.fig_ncols)
        fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
        if len(self.variable_names)>1: axs = axs.flatten()
        for i, var in enumerate(self.variable_names):
            ax = axs[i] if len(self.variable_names)>1 else axs
            self.variable_metric_boxplot(metric, var, kind, exp_var_bins, ax, fig_ncols)
            if add_title: ax.set_title(var)
        if num_variables%self.fig_ncols!=0 and len(self.variable_names)>1: fig.delaxes(axs.flatten()[-1])
        if fig_ncols is not None: 
            self.fig_ncols = previous_ncols
            if fig_ncols > 2: 
                for ax in axs:
                    for tick in ax.get_xticklabels():
                        tick.set_rotation(45)
        fig.tight_layout()
        if save_name is not None: savefigure(fig, save_name) 
        plt.show()
    
    def variable_calibration_plot(self, metric, variable, k=100, log_bins=False, ax=None, hi_bound=np.inf, palette=None, show_legend=True, show_cbar=True, **kwargs):
        bin_type = "log" if log_bins else "linear"
        if ax is None:
            fig, ax = plt.subplots()
        var_idx = self.variable_names.index(variable)
        assert metric in ["uce", "ence", "auce"]
        if metric == "uce": cols = ["bin variance", "bin mse", self.exp_var_name]
        elif metric == "auce": cols = ["expected accuracy", "empirical accuracy", self.exp_var_name]
        else: cols = ["binned predictive uncertainty", "binned RMSE", self.exp_var_name]
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
            cmap = sns.color_palette("bwr")
            r, b = cmap[0],cmap[-1]
            try: ax.get_legend().remove()
            except: pass
            # ccdf[cols[-1]] = ccdf[cols[-1]].astype("float")
            # Compute max
            argmax_expvar = np.argmax(self.exp_vars)
            max_rcu = self.rcus[argmax_expvar]
            max_rcu = max_rcu.upsample(k, bin_type=bin_type)
            xc, yc = max_rcu.get_calibration_curve(metric)
            if metric != "auce": xc = xc[var_idx]; yc = yc[var_idx]
            mask = xc<hi_bound
            xc, yc = xc[mask], yc[mask]
            ax.plot(xc, yc, color=b, label=f"maximum: {self.exp_vars[argmax_expvar]:.2f}%", alpha=1)
            # Compute min
            argmin_expvar = np.argmin(self.exp_vars)
            min_rcu = self.rcus[argmin_expvar]
            min_rcu = min_rcu.upsample(k, bin_type=bin_type)
            xc, yc = min_rcu.get_calibration_curve(metric)
            if metric != "auce": xc = xc[var_idx]; yc = yc[var_idx]
            mask = xc<hi_bound
            xc, yc = xc[mask], yc[mask]
            ax.plot(xc, yc, color=r, label=f"minimum: {self.exp_vars[argmin_expvar]:.2f}%", alpha=1)
            handles, labels = ax.get_legend_handles_labels()
            ax.legend([handle for i,handle in enumerate(handles[-2:])],
                    [label for i,label in enumerate(labels[-2:])],
                    title="average cloud probability",
                    loc= 'best')
        else:
            ax.legend(loc="upper right")
        if show_cbar:
            sm = plt.cm.ScalarMappable(cmap="bwr", norm=plt.Normalize(min(self.exp_vars), max(self.exp_vars)))
            ax.figure.colorbar(sm, ax=ax)
        # ax.vlines(x=xc, ymin=lo, ymax=hi/5, color="black", ls="dashed")
        return ax
    
    def calibration_plot(self, metric, k=100, log_bins=False, hi_bounds=np.inf, figsize=(12, 18), fig_ncols=None, save_name=None, variables=None, add_title=True, **kwargs):
        if not isinstance(hi_bounds, list): hi_bounds = [hi_bounds for _ in range(len(self.variable_names))]
        if variables is None:
            variables = self.variable_names 
        num_variables = len(variables)
        if fig_ncols is not None:
            previous_ncols = self.fig_ncols
            self.fig_ncols = fig_ncols
        ncols = self.fig_ncols
        nrows = 1 if self.fig_ncols>=num_variables else math.ceil(num_variables/self.fig_ncols)
        fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
        if len(variables)>1: axs = axs.flatten()
        for i, var in enumerate(variables):
            ax = axs[i] if len(variables)>1 else axs
            self.variable_calibration_plot(metric, var, k=k, log_bins=log_bins, ax=ax, hi_bound=hi_bounds[i], show_cbar=i==len(variables)-1, **kwargs)
            if add_title: ax.set_title(var)
        if kwargs.get("invite", None) == True:
            axs[0].set_xlabel("")
            axs[1].set_xlabel("")
            axs[3].set_xlabel("")
            axs[4].set_xlabel("")
            #
            axs[1].set_ylabel("")
            axs[2].set_ylabel("")
            axs[3].set_ylabel("")
            axs[4].set_ylabel("")
            #
            axs[0].legend([],[],frameon=False)
            axs[1].legend([],[],frameon=False)
            axs[3].legend([],[],frameon=False)
            axs[4].legend([],[],frameon=False)
            #
            sns.move_legend(axs[2], "upper center", ncol=2, title="average cloud probability", fontsize="small", bbox_to_anchor=(0.5,1.3))
        # fig.suptitle(metric+" calibration plot")
        if num_variables%self.fig_ncols!=0: fig.delaxes(axs.flatten()[-1])
        if fig_ncols is not None: self.fig_ncols = previous_ncols
        fig.tight_layout()
        if save_name is not None: savefigure(fig, save_name)
        plt.show()

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

def showSplit(gt_path, split_mask, islices=None, jslices=None, 
              patch_colors=None, save_name=None):
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
    if islices is not None:
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
            figsize=(12, 4), split_mask=None, save_name=None, color="g", nrows=1,
            show_avg_cp=False):
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
        cp = loadRaster(img_path, bands=-1)
        if islice is not None and jslice is not None: 
            if not isinstance(islice, list): islice = [islice]
            if not isinstance(jslice, list): jslice = [jslice]
            if isinstance(color, list): assert len(color)==len(islice)
            assert len(islice)==len(jslice)
            for k, (islc, jslc) in enumerate(zip(islice, jslice)):
                if not draw_bbox:
                    cp = cp[islc, jslc]
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
        if show_avg_cp: title +="\naverage Cp: {0:.2f}%".format(cp.mean())
        ax.set_title(title)
        ax.set_axis_off()
        i += 1
    plt.tight_layout()
    if save_name is not None: savefigure(fig, save_name)
    plt.show()

def showConditionCloudDistribution(
    dirs,
    titles,
    s2repr_dirs,
    islice=None,
    jslice=None,
    save_name=None
):
    fig, axs = plt.subplots(ncols=len(dirs), nrows=2, figsize=(12, 6))
    for j, d in enumerate(dirs):
        img_path = getPaths(d, s2repr_dirs=s2repr_dirs, returns=["img"])
        cp_global = loadRaster(img_path, bands=-1, islice=None, jslice=None).flatten()
        cp_sliced = loadRaster(img_path, bands=-1, islice=islice, jslice=jslice).flatten()
        sns.histplot(x=cp_global, bins=15, binrange=(0,100), ax=axs[0,j])
        axs[0,j].set_title(f"{titles[j]}\nmean={np.nanmean(cp_global):.3f}")
        if j==0: axs[0,j].set_ylabel("Global\ncount")
        else: axs[0,j].set_ylabel("")
        axs[0,j].set_xlabel("cloud probability")
        axs[0,j].set_yscale("log")
        axs[0,j].set_xlim(0,100)
        axs[0,j].set_ylim(top=1e7)
        sns.histplot(x=cp_sliced, bins=15, binrange=(0,100), ax=axs[1,j])
        axs[1,j].set_title(f"mean={np.nanmean(cp_sliced):.3f}")
        if j==0: axs[1,j].set_ylabel("Sliced\ncount")
        else: axs[1,j].set_ylabel("")
        axs[1,j].set_xlabel("cloud probability")
        axs[1,j].set_yscale("log")
        axs[1,j].set_xlim(0,100)
        axs[1,j].set_ylim(top=1e5)
    plt.tight_layout()
    if save_name is not None: savefigure(fig, save_name)
    plt.show()

def showEmpiricalSelectionRejectionProbs(
    cp_df,
    empirical_cp_threshold,
    interpolate=True,
    figsize=(12,8),
    save_name=None
):
    if interpolate:
        cp_df = cp_df.interpolate(limit_direction="both")
    fig = plt.figure(figsize=figsize)
    ax = plt.gca()
    sns.lineplot(data=cp_df, x="cp", y="selection_probability", color="tab:blue", label="selection", ax=ax)
    sns.lineplot(data=cp_df, x="cp", y="rejection_probability", color="tab:orange", label="rejection", ax=ax)
    ax.vlines(empirical_cp_threshold, 0, 1, linestyle="dashed", color="black", 
            label=f"empirical threshold (cp={empirical_cp_threshold}%)")
    ax.set_ylabel("empirical probability")
    ax.set_xlabel("cloud probability")
    ax.legend()
    plt.tight_layout()
    if save_name is not None: savefigure(fig, save_name)
    plt.show()

def computeValidCenterFraction(
    dirs, 
    titles, 
    s2repr_dirs, 
    empirical_cp_threshold,
    islice=None, 
    jslice=None,
    figsize=(10, 4),
    save_name=None,
):
    fig, axs = plt.subplots(ncols=len(dirs), nrows=1, figsize=figsize)
    data = {"condition": [], "sliced selected fraction (%)": [], "complete selected fraction (%)": []}
    for i, (d, t) in enumerate(zip(dirs, titles)):
        img_path  = getPaths(d, s2repr_dirs=s2repr_dirs, returns=["img"])
        cp = loadRaster(img_path, bands=-1, islice=None, jslice=None, dtype="float").reshape(-1)
        slice_cp = loadRaster(img_path, bands=-1, islice=islice, jslice=jslice, dtype="float")
        slice_cp_flat = slice_cp.reshape(-1)
        data["condition"].append(t)
        for vals, col in zip([cp, slice_cp_flat], ["sliced selected fraction (%)", "complete selected fraction (%)"]):
            data[col].append((vals<=empirical_cp_threshold).astype("int").sum()/vals.shape[0]*100)
        rgb = loadRaster(img_path, bands=[4,3,2], islice=islice, jslice=jslice, transpose_order=(1,2,0), clip_range=(100, 2000))
        axs[i].imshow(rgb)
        cp_mask = (slice_cp<=empirical_cp_threshold).astype("float")
        cp_mask[cp_mask==0] = -1
        cp_mask[cp_mask==1] = 1
        sns.heatmap(cp_mask, cmap="bwr_r", alpha=0.65, cbar=False, ax=axs[i], vmin=-1, vmax=1)
        frac = float(pd.DataFrame(data).query(f"condition == '{t}'")["sliced selected fraction (%)"])
        axs[i].set_title(f"{t}\n({frac:.2f}% selected)")
        axs[i].set_axis_off()
    plt.tight_layout()
    if save_name is not None: savefigure(fig, save_name)
    plt.show()
    return pd.DataFrame(data)


def showPairedBandsHistograms(d1, d2, s2repr_dirs, islice=None, jslice=None, save_name=None):
    nc_path = getPaths(d1, s2repr_dirs=s2repr_dirs, returns=["img"])
    vc_path = getPaths(d2, s2repr_dirs=s2repr_dirs, returns=["img"])
    nc = loadRaster(nc_path, islice=islice, jslice=jslice, bands=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]).reshape(12,-1)
    vc = loadRaster(vc_path, islice=islice, jslice=jslice, bands=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]).reshape(12,-1)
    fig, axs = plt.subplots(ncols=4, nrows=3, figsize=(15,7))
    axs = axs.flatten()
    for i, ax in enumerate(axs):
        df = pd.DataFrame({
            "values": nc[i].tolist()+vc[i].tolist(),
            "conditions": ["no cloud" for _ in nc[i]]+["very cloudy" for _ in vc[i].tolist()],
        })
        sns.histplot(data=df, x="values", hue="conditions", ax=ax, binwidth=30)
        ax.set_title(f"Sentinel-2 band {i+1}")
        ax.set_xlabel("")
        ax.set_yscale("log")
        if i not in [0,4,8]: ax.set_ylabel("")
        if i<11: ax.legend([],[],frameon=False)
    plt.tight_layout()
    if save_name is not None: savefigure(fig, save_name)
    plt.show()

def showAllSentinel2Bands(d1, d2, s2repr_dirs, islice, jslice, save_name, bands=[6,7,8,9,11], figsize=(12,5)):
    nc_path = getPaths(d1, s2repr_dirs=s2repr_dirs, returns=["img"])
    vc_path = getPaths(d2, s2repr_dirs=s2repr_dirs, returns=["img"])
    nc = loadRaster(nc_path, islice=islice, jslice=jslice, bands=bands)
    vc = loadRaster(vc_path, islice=islice, jslice=jslice, bands=bands)
    fig, axs = plt.subplots(ncols=len(bands), nrows=2, figsize=figsize)
    for i, b in enumerate(bands):
        vmin, vmax = multiMinMax(nc[i], vc[i])
        sns.heatmap(nc[i], ax=axs[0,i], cbar=False, vmin=vmin, vmax=vmax)
        if i==0: axs[0,i].set_ylabel("no cloud")
        axs[0,i].set_xticks([])
        axs[0,i].set_yticks([])
        axs[0,i].set_title(f"Sentinel-2\nband {b}")
        sns.heatmap(vc[i], ax=axs[1,i], cbar=False, vmin=vmin, vmax=vmax)
        if i==0: axs[1,i].set_ylabel("very cloudy")
        axs[1,i].set_xticks([])
        axs[1,i].set_yticks([])
    plt.tight_layout()
    if save_name is not None: savefigure(fig, save_name)
    plt.show()

def showClosestSentinel1Bands(d, s1_dir, islice, jslice, save_name):
    s2_date = datetime.strptime(
        d.split("/")[-1].split("_")[-1].split("T")[0],
        "%Y%m%d"
    )
    delta = float("inf")
    s1_path = None 
    for this_s1_path in Path(s1_dir).glob("*.tif"):
        s1_name = this_s1_path.name
        this_s1_date = datetime.strptime(
            s1_name.split("_")[5].split("T")[0],
            "%Y%m%d"
        )
        if s1_path is None or abs(this_s1_date-s2_date)<delta: 
            s1_path = this_s1_path
            delta = abs(this_s1_date-s2_date)
    s1_bands = loadRaster(s1_path, islice=islice, jslice=jslice, bands=[1,2])
    fig, axs = plt.subplots(ncols=2, nrows=1, figsize=(12,5))
    axs = axs.flatten()
    vmaxs = [200, 400]
    for i, ax in enumerate(axs):
        sns.heatmap(s1_bands[i], ax=ax, vmax=vmaxs[i], cbar=False)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"Sentinel-1 band: {i+1}")
    plt.tight_layout()
    if save_name is not None: savefigure(fig, save_name)
    plt.show()

def showDistanceToTrainsetExpectation(
    dirs,
    titles,
    s2repr_dirs,
    variable_indexes,
    trainset_means,
    variable_names,
    islice=None,
    jslice=None,
    figsize=(12, 6),
    save_name=None,
    ncols=None, nrows=None
):
    fig, axs = plt.subplots(ncols=ncols, nrows=nrows, figsize=figsize)
    axs = axs.flatten()
    df = pd.DataFrame()
    for d, c in zip(dirs, titles):
        mean_path  = getPaths(d, s2repr_dirs=s2repr_dirs, returns=["mean"])
        mean = loadRaster(mean_path, bands=variable_indexes, islice=islice, jslice=jslice).reshape(len(variable_names), -1)
        mask = ~np.isnan(mean).all(0)
        mean = mean[:,mask]
        for i, (variable_name, var_index) in enumerate(zip(variable_names, variable_indexes)):
            dfc = pd.DataFrame({
                "mean": mean[i]-trainset_means[var_index-1],
                "condition": c,
                "variable": variable_name
            })
            dfc = dfc.groupby(by=["condition", "variable"]).agg(
                avg_mean=("mean", lambda x: np.nanmean(x)),
                std_mean=("mean", lambda x: np.nanstd(x)),
            )
            df = pd.concat([df, dfc], axis=0)
    for i, (ax, variable_name) in enumerate(zip(axs, variable_names)):
        sub = df.query(f"variable == '{variable_name}'")
        sns.lineplot(data=sub, x="condition", y="avg_mean", ax=ax)
        color = ax.get_lines()[-1].get_c()
        ax.fill_between(range(0,len(titles)), 
                        sub.avg_mean+sub.std_mean, 
                        sub.avg_mean-sub.std_mean,
                        color=color, alpha=0.3)
        ax.set_title(variable_name)
        ylim = max(abs(l) for l in ax.get_ylim())
        ax.set_ylim(-ylim, ylim)
        ax.hlines(0, xmin=0, xmax=len(titles)-1, linestyle="dotted", color="black")
        ax.set_xlabel("")
        if i==0: ax.set_ylabel("mean distance to\ntrain set expectation") 
        else: ax.set_ylabel("")
        for tick in ax.get_xticklabels():
            tick.set_rotation(45)
    plt.tight_layout()
    if save_name is not None: savefigure(fig, save_name)
    plt.show()

def showMeanDiffWithExpectation(
    d,
    trainset_means,
    variable_names,
    islice=None,
    jslice=None,
    figsize=(12,2),
    save_name=None
):
    trainset_means = np.array(trainset_means).reshape(-1,1,1)
    fig, axs = plt.subplots(ncols=len(variable_names), nrows=1, figsize=figsize)
    mean_path = getPaths(d, returns=["mean"])
    diffs = loadRaster(mean_path, bands=None, islice=islice, jslice=jslice)-trainset_means
    for i, variable_name in enumerate(variable_names):
        bound = np.nanmax(np.abs(diffs[i]))
        sns.heatmap(diffs[i], cmap="bwr", vmin=-bound, vmax=bound, ax=axs[i])
        axs[i].set_title(variable_name+f"\n(E[Y]={trainset_means[i,0,0]:.3f})")
        axs[i].set_axis_off()
    plt.tight_layout()
    if  save_name is not None: savefigure(fig, save_name)
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
    
def showPredictionMaps(dirs, titles, variable_index, variable_name, s2repr_dirs, gt_dir, 
                        shapefile_paths, islice=None, jslice=None, normalize=False, save_name=None,
                        predictive_uncertainty_bounds=None, rerror_bounds=None, cerror_bounds=None,
                        figsize=(15,22.75)):
    fig, axs = plt.subplots(ncols=len(titles), nrows=7, figsize=figsize)
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
        if normalize:
            gt, mean, predictive_std = multiNorm2d(gt, mean, predictive_std)
            vmin, vmax = 0, 1
        else:
            vmin, vmax = multiMinMax(gt, mean, predictive_std)
        dvmin, dvmax = vmin-vmax, vmax-vmin
        rerror = mean-gt
        cerror = np.abs(rerror)-predictive_std
        pubound = vmax if predictive_uncertainty_bounds is None else predictive_uncertainty_bounds[variable_index-1]
        rbound = np.nanmax(np.abs(rerror)) if rerror_bounds is None else rerror_bounds[variable_index-1]
        cbound = np.nanmax(np.abs(cerror)) if cerror_bounds is None else cerror_bounds[variable_index-1]
        axs[0,i].imshow(rgb)
        if i==0: axs[0,i].set_ylabel(f"RGB")
        sns.heatmap(gt, ax=axs[1,i], vmin=vmin, vmax=vmax, cbar=i==len(titles)-1)
        if i==0: axs[1,i].set_ylabel(f"ground truth\n({gt_date.strftime('%d.%m.%Y')})")
        sns.heatmap(mean, ax=axs[2,i], vmin=vmin, vmax=vmax, cbar=i==len(titles)-1)
        if i==0: axs[2,i].set_ylabel(f"predicted\nmean")
        sns.heatmap(rerror, ax=axs[3,i], cmap="bwr", vmin=-rbound, vmax=rbound, cbar=i==len(titles)-1)
        if i==0: axs[3,i].set_ylabel(f"regression\nresiduals")
        sns.heatmap(predictive_std, ax=axs[4,i], vmin=0, vmax=pubound, cbar=i==len(titles)-1)
        if i==0: axs[4,i].set_ylabel(f"predictive\nuncertainty")
        sns.heatmap(np.abs(rerror), ax=axs[5,i], vmin=0, vmax=pubound, cbar=i==len(titles)-1)
        if i==0: axs[5,i].set_ylabel(f"absolute\nregression residuals")
        sns.heatmap(cerror, ax=axs[6,i], cmap="bwr", vmin=-cbound, vmax=cbound, cbar=i==len(titles)-1)
        if i==0: axs[6,i].set_ylabel(f"calibration\nresiduals")
    for ax in axs.flatten(): 
        ax.set_xticks([])
        ax.set_yticks([])
    for i, d, in enumerate(dirs):
        axs[0,i].set_title(f'{titles[i]}\n({datetime.strptime(Path(d).name.split("_")[1].split("T")[0], "%Y%m%d").strftime("%d.%m.%Y")})')
    fig.suptitle(variable_name)
    plt.tight_layout()
    if save_name is not None: savefigure(fig, save_name)
    plt.show()

def showTransposedPredictionMaps(dirs, titles, variable_index, variable_name, s2repr_dirs, gt_dir, 
                        shapefile_paths, islice=None, jslice=None, normalize=False, save_name=None,
                        predictive_uncertainty_bounds=None, rerror_bounds=None, cerror_bounds=None,
                        figsize=(15,22.75)):
    fig, axs = plt.subplots(nrows=len(titles), ncols=7, figsize=figsize)
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
        if normalize:
            gt, mean, predictive_std = multiNorm2d(gt, mean, predictive_std)
            vmin, vmax = 0, 1
        else:
            vmin, vmax = multiMinMax(gt, mean, predictive_std)
        dvmin, dvmax = vmin-vmax, vmax-vmin
        rerror = mean-gt
        cerror = np.abs(rerror)-predictive_std
        pubound = vmax if predictive_uncertainty_bounds is None else predictive_uncertainty_bounds[variable_index-1]
        rbound = np.nanmax(np.abs(rerror)) if rerror_bounds is None else rerror_bounds[variable_index-1]
        cbound = np.nanmax(np.abs(cerror)) if cerror_bounds is None else cerror_bounds[variable_index-1]
        if i==len(titles)-1:
            kwargs = dict(cbar=True, cbar_kws={"location": "bottom",
              "orientation": "horizontal",
              "use_gridspec": True,
              "pad": -0.1,
              "anchor": (0.5,-0.90)})
        else:
            kwargs = dict(cbar=False)
        axs[i,0].imshow(rgb)
        if i==0: axs[i,0].set_title(f"RGB")
        sns.heatmap(gt, ax=axs[i,1], vmin=vmin, vmax=vmax, **kwargs)
        if i==0: axs[i,1].set_title(f"ground truth\n({gt_date.strftime('%d.%m.%Y')})")
        sns.heatmap(mean, ax=axs[i,2], vmin=vmin, vmax=vmax, **kwargs)
        if i==0: axs[i,2].set_title(f"predicted\nmean")
        sns.heatmap(rerror, ax=axs[i,3], cmap="bwr", vmin=-rbound, vmax=rbound, **kwargs)
        if i==0: axs[i,3].set_title(f"regression\nresiduals")
        sns.heatmap(predictive_std, ax=axs[i,4], vmin=0, vmax=pubound, **kwargs)
        if i==0: axs[i,4].set_title(f"predictive\nuncertainty")
        sns.heatmap(np.abs(rerror), ax=axs[i,5], vmin=0, vmax=pubound, **kwargs)
        if i==0: axs[i,5].set_title(f"absolute\nregression residuals")
        sns.heatmap(cerror, ax=axs[i,6], cmap="bwr", vmin=-cbound, vmax=cbound, **kwargs)
        if i==0: axs[i,6].set_title(f"calibration\nresiduals")
    for ax in axs.flatten(): 
        ax.set_xticks([])
        ax.set_yticks([])
    for i, d, in enumerate(dirs):
        axs[i,0].set_ylabel(f'{titles[i]}\n({datetime.strptime(Path(d).name.split("_")[1].split("T")[0], "%Y%m%d").strftime("%d.%m.%Y")})')
    fig.suptitle(variable_name)
    plt.tight_layout()
    if save_name is not None: savefigure(fig, save_name)
    plt.show()

def showRegressionMaps(dirs, titles, variable_index, variable_name, s2repr_dirs, gt_dir, 
                        shapefile_paths, islice=None, jslice=None, normalize=False, save_name=None,
                        rerror_bounds=None, figsize=(15,13), **kwargs):
    fig, axs = plt.subplots(ncols=len(titles), nrows=4, figsize=figsize)
    for i, d in enumerate(dirs):
        dir_name = d.split("/")[-1]
        pid = dir_name.split("_")[0]
        gt_date = compute_gt_date(pid, shapefile_paths)
        img_path, mean_path, gt_path = getPaths(d, s2repr_dirs, gt_dir, returns=["img","mean","gt"])
        # load rasters
        rgb = loadRaster(img_path, bands=[4,3,2], transpose_order=(1,2,0), clip_range=(100, 2000), islice=islice, jslice=jslice)
        gt = loadRaster(gt_path, bands=variable_index, islice=islice, jslice=jslice, set_nan_mask=True)
        if variable_index in [3,5]: gt /= 100 # Cover/Dens normalization!!
        mean = loadRaster(mean_path, bands=variable_index, islice=islice, jslice=jslice)
        if normalize:
            gt, mean = multiNorm2d(gt, mean)
            vmin, vmax = 0, 1
        else:
            vmin, vmax = multiMinMax(gt, mean)
        rerror = mean-gt
        rbound = np.nanmax(np.abs(rerror)) if rerror_bounds is None else rerror_bounds[variable_index-1]
        axs[0,i].imshow(rgb)
        if i==0: axs[0,i].set_ylabel(f"RGB")
        sns.heatmap(gt, ax=axs[1,i], vmin=vmin, vmax=vmax, cbar=i==len(titles)-1)
        if i==0: axs[1,i].set_ylabel(f"ground truth\n({gt_date.strftime('%d.%m.%Y')})")
        sns.heatmap(mean, ax=axs[2,i], vmin=vmin, vmax=vmax, cbar=i==len(titles)-1)
        if i==0: axs[2,i].set_ylabel(f"predicted\nmean")
        sns.heatmap(rerror, ax=axs[3,i], cmap="bwr", vmin=-rbound, vmax=rbound, cbar=i==len(titles)-1)
        if i==0: axs[3,i].set_ylabel(f"regression\nresiduals")
    for ax in axs.flatten(): 
        ax.set_xticks([])
        ax.set_yticks([])
    for i, d, in enumerate(dirs):
        axs[0,i].set_title(f'{titles[i]}\n({datetime.strptime(Path(d).name.split("_")[1].split("T")[0], "%Y%m%d").strftime("%d.%m.%Y")})')
    fig.suptitle(variable_name)
    plt.tight_layout()
    if save_name is not None: savefigure(fig, save_name)
    plt.show()

def showUsabilityMaps(dirs, titles, variable_index, variable_name, s2repr_dirs, gt_dir, 
                        shapefile_paths, islice=None, jslice=None, normalize=False, save_name=None,
                        figsize=(15,6.5), predictive_uncertainty_bounds=None, **kwargs):
    fig, axs = plt.subplots(ncols=len(titles), nrows=2, figsize=figsize)
    for i, d in enumerate(dirs):
        dir_name = d.split("/")[-1]
        pid = dir_name.split("_")[0]
        gt_date = compute_gt_date(pid, shapefile_paths)
        img_path, variance_path = getPaths(d, s2repr_dirs, gt_dir, returns=["img","variance"])
        # load rasters
        rgb = loadRaster(img_path, bands=[4,3,2], transpose_order=(1,2,0), clip_range=(100, 2000), islice=islice, jslice=jslice)
        predictive_std = loadRaster(variance_path, bands=variable_index, islice=islice, jslice=jslice, elementwise_fn=np.sqrt)
        if normalize:
            predictive_std, _ = multiNorm2d(predictive_std, predictive_std)
            vmin, vmax = 0, 1
        else:
            vmin, vmax = multiMinMax(predictive_std, predictive_std)
        pubound = vmax if predictive_uncertainty_bounds is None else predictive_uncertainty_bounds[variable_index-1]
        axs[0,i].imshow(rgb)
        if i==0: axs[0,i].set_ylabel(f"RGB")
        sns.heatmap(predictive_std, ax=axs[1,i], vmin=0, vmax=pubound, cbar=i==len(titles)-1)
        if i==0: axs[1,i].set_ylabel(f"predictive\nuncertainty")
    for ax in axs.flatten(): 
        ax.set_xticks([])
        ax.set_yticks([])
    for i, d, in enumerate(dirs):
        axs[0,i].set_title(f'{titles[i]}\n({datetime.strptime(Path(d).name.split("_")[1].split("T")[0], "%Y%m%d").strftime("%d.%m.%Y")})')
    fig.suptitle(variable_name)
    plt.tight_layout()
    if save_name is not None: savefigure(fig, save_name)
    plt.show()

def showHybridCalibrationMaps(dirs, titles, variable_indexes, variable_names, s2repr_dirs, gt_dir, 
                        shapefile_paths, islice=None, jslice=None, normalize=False, save_name=None,
                        cerror_bounds=None, figsize=(15,22.75), **kwargs):
    fig, axs = plt.subplots(nrows=1+len(variable_names), ncols=len(titles), figsize=figsize)
    for i, d in enumerate(dirs):
        dir_name = d.split("/")[-1]
        pid = dir_name.split("_")[0]
        gt_date = compute_gt_date(pid, shapefile_paths)
        img_path, mean_path, gt_path, variance_path = getPaths(d, s2repr_dirs, gt_dir, returns=["img","mean","gt", "variance"])
        # load rasters
        rgb = loadRaster(img_path, bands=[4,3,2], transpose_order=(1,2,0), clip_range=(100, 2000), islice=islice, jslice=jslice)
        gts = loadRaster(gt_path, bands=variable_indexes, islice=islice, jslice=jslice, set_nan_mask=True)
        means = loadRaster(mean_path, bands=variable_indexes, islice=islice, jslice=jslice)
        predictive_stds = loadRaster(variance_path, bands=variable_indexes, islice=islice, jslice=jslice, elementwise_fn=np.sqrt)
        axs[0,i].imshow(rgb)
        if i==0: axs[0,i].set_ylabel(f"RGB")
        for j, (variable_name, variable_index) in enumerate(zip(variable_names, variable_indexes)):
            if variable_name in ["Dens", "Cover"]: gts[j] /= 100
            k = j+1
            rerror = means[j]-gts[j]
            cerror = np.abs(rerror)-predictive_stds[j]
            if normalize:
                if cerror_bounds is None: raise NotImplementedError()
                else: 
                    cerror = norm2d(cerror, -cerror_bounds[variable_index-1],cerror_bounds[variable_index-1], -1, 1)
                    vmin,vmax = -1, 1
            else: vmin,vmax = -cerror_bounds[variable_index-1],cerror_bounds[variable_index-1]
            sns.heatmap(cerror, ax=axs[k,i], cmap="bwr", cbar=i==len(titles)-1,  vmin=vmin, vmax=vmax)
            if i==0: axs[k,i].set_ylabel(variable_name)
    for ax in axs.flatten(): 
        ax.set_xticks([])
        ax.set_yticks([])
    for i, d, in enumerate(dirs):
        axs[0,i].set_title(f'{titles[i]}\n({datetime.strptime(Path(d).name.split("_")[1].split("T")[0], "%Y%m%d").strftime("%d.%m.%Y")})')
    fig.suptitle(variable_name)
    plt.tight_layout()
    if save_name is not None: savefigure(fig, save_name)
    plt.show()

def showCalibrationMaps(dirs, titles, variable_index, variable_name, s2repr_dirs, gt_dir, 
                        shapefile_paths, islice=None, jslice=None, normalize=False, save_name=None,
                        predictive_uncertainty_bounds=None, cerror_bounds=None,
                        figsize=(15,13), **kwargs):
    fig, axs = plt.subplots(ncols=len(titles), nrows=4, figsize=figsize)
    for i, d in enumerate(dirs):
        dir_name = d.split("/")[-1]
        pid = dir_name.split("_")[0]
        gt_date = compute_gt_date(pid, shapefile_paths)
        img_path, mean_path, gt_path, variance_path = getPaths(d, s2repr_dirs, gt_dir, returns=["img","mean","gt", "variance"])
        # load rasters
        rgb = loadRaster(img_path, bands=[4,3,2], transpose_order=(1,2,0), clip_range=(100, 2000), islice=islice, jslice=jslice)
        gt = loadRaster(gt_path, bands=variable_index, islice=islice, jslice=jslice, set_nan_mask=True)
        if variable_index in [3,5]: gt /= 100 # Cover/Dens normalization!!
        mean = loadRaster(mean_path, bands=variable_index, islice=islice, jslice=jslice)
        predictive_std = loadRaster(variance_path, bands=variable_index, islice=islice, jslice=jslice, elementwise_fn=np.sqrt)
        if normalize:
            gt, mean = multiNorm2d(gt, mean, predictive_std)
            vmin, vmax = 0, 1
        else:
            vmin, vmax = multiMinMax(gt, mean, predictive_std)
        rerror = mean-gt
        cerror = np.abs(rerror)-predictive_std
        pubound = vmax if predictive_uncertainty_bounds is None else predictive_uncertainty_bounds[variable_index-1]
        cbound = np.nanmax(np.abs(cerror)) if cerror_bounds is None else cerror_bounds[variable_index-1]
        axs[0,i].imshow(rgb)
        if i==0: axs[0,i].set_ylabel(f"RGB")
        sns.heatmap(predictive_std, ax=axs[1,i], vmin=0, vmax=pubound, cbar=i==len(titles)-1)
        if i==0: axs[1,i].set_ylabel(f"predictive\nuncertainty")
        sns.heatmap(np.abs(rerror), ax=axs[2,i], vmin=0, vmax=pubound, cbar=i==len(titles)-1)
        if i==0: axs[2,i].set_ylabel(f"absolute\nregression residuals")
        sns.heatmap(cerror, ax=axs[3,i], cmap="bwr", vmin=-cbound, vmax=cbound, cbar=i==len(titles)-1)
        if i==0: axs[3,i].set_ylabel(f"calibration\nresiduals")
    for ax in axs.flatten(): 
        ax.set_xticks([])
        ax.set_yticks([])
    for i, d, in enumerate(dirs):
        axs[0,i].set_title(f'{titles[i]}\n({datetime.strptime(Path(d).name.split("_")[1].split("T")[0], "%Y%m%d").strftime("%d.%m.%Y")})')
    fig.suptitle(variable_name)
    plt.tight_layout()
    if save_name is not None: savefigure(fig, save_name)
    plt.show()

def showQuasiCalibrationMaps(dirs, titles, variable_index, variable_name, s2repr_dirs, gt_dir, 
                        shapefile_paths, islice=None, jslice=None, normalize=False, save_name=None,
                        predictive_uncertainty_bounds=None, cerror_bounds=None, rerror_bounds=None,
                        figsize=(15,13), **kwargs):
    fig, axs = plt.subplots(ncols=len(titles), nrows=4, figsize=figsize)
    for i, d in enumerate(dirs):
        dir_name = d.split("/")[-1]
        pid = dir_name.split("_")[0]
        gt_date = compute_gt_date(pid, shapefile_paths)
        img_path, mean_path, gt_path, variance_path = getPaths(d, s2repr_dirs, gt_dir, returns=["img","mean","gt", "variance"])
        # load rasters
        rgb = loadRaster(img_path, bands=[4,3,2], transpose_order=(1,2,0), clip_range=(100, 2000), islice=islice, jslice=jslice)
        gt = loadRaster(gt_path, bands=variable_index, islice=islice, jslice=jslice, set_nan_mask=True)
        if variable_index in [3,5]: gt /= 100 # Cover/Dens normalization!!
        mean = loadRaster(mean_path, bands=variable_index, islice=islice, jslice=jslice)
        predictive_std = loadRaster(variance_path, bands=variable_index, islice=islice, jslice=jslice, elementwise_fn=np.sqrt)
        if normalize:
            gt, mean = multiNorm2d(gt, mean, predictive_std)
            vmin, vmax = 0, 1
        else:
            vmin, vmax = multiMinMax(gt, mean, predictive_std)
        rerror = mean-gt
        cerror = np.abs(rerror)-predictive_std
        pubound = vmax if predictive_uncertainty_bounds is None else predictive_uncertainty_bounds[variable_index-1]
        rbound = np.nanmax(np.abs(rerror)) if rerror_bounds is None else rerror_bounds[variable_index-1]
        cbound = np.nanmax(np.abs(cerror)) if cerror_bounds is None else cerror_bounds[variable_index-1]
        axs[0,i].imshow(rgb)
        if i==0: axs[0,i].set_ylabel(f"RGB")
        sns.heatmap(predictive_std, ax=axs[1,i], vmin=0, vmax=pubound, cbar=i==len(titles)-1)
        if i==0: axs[1,i].set_ylabel(f"predictive\nuncertainty")
        sns.heatmap(rerror, ax=axs[2,i], cmap="bwr", vmin=-rbound, vmax=rbound, cbar=i==len(titles)-1)
        if i==0: axs[2,i].set_ylabel(f"regression\nresiduals")
        sns.heatmap(cerror, ax=axs[3,i], cmap="bwr", vmin=-cbound, vmax=cbound, cbar=i==len(titles)-1)
        if i==0: axs[3,i].set_ylabel(f"calibration\nresiduals")
    for ax in axs.flatten(): 
        ax.set_xticks([])
        ax.set_yticks([])
    for i, d, in enumerate(dirs):
        axs[0,i].set_title(f'{titles[i]}\n({datetime.strptime(Path(d).name.split("_")[1].split("T")[0], "%Y%m%d").strftime("%d.%m.%Y")})')
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
    mxbounds = [np.inf, -np.inf]
    mybound = -np.inf
    uxbound = -np.inf
    uybound = -np.inf
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
        # compute bounds
        mxmin, mxmax = axs[i,0].get_xlim()
        if mxmin < mxbounds[0]: mxbounds[0] = mxmin
        if mxmax > mxbounds[1]: mxbounds[1] = mxmax
        mymax = axs[i,0].get_ylim()[1]
        if mymax > mybound: mybound = mymax
        uxmax = axs[i,1].get_xlim()[1]
        if uxmax > uxbound: uxbound = uxmax
        uymax = axs[i,1].get_ylim()[1]
        if uymax > uybound: uybound = uymax
    for i, d, in enumerate(dirs):
        axs[i,0].set_ylabel(f'{titles[i]}\nCount')
        axs[i,0].set_xlim(*mxbounds)
        axs[i,0].set_ylim(0, mybound)
        axs[i,1].set_xlim(0, uxbound)
        axs[i,1].set_ylim(0, uybound)
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
    cxbounds = [np.inf, -np.inf]
    cybound = -np.inf
    rxbounds = [np.inf, -np.inf]
    rybound = -np.inf
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
        # compute bounds
        rxmin, rxmax = axs[i,0].get_xlim()
        if rxmin < rxbounds[0]: rxbounds[0] = rxmin
        if rxmax > rxbounds[1]: rxbounds[1] = rxmax
        rymax = axs[i,0].get_ylim()[1]
        if rymax > rybound: rybound = rymax
        cxmin, cxmax = axs[i,1].get_xlim()
        if cxmin < cxbounds[0]: cxbounds[0] = cxmin
        if cxmax > cxbounds[1]: cxbounds[1] = cxmax
        cymax = axs[i,1].get_ylim()[1]
        if cymax > cybound: cybound = cymax
    for i, d, in enumerate(dirs):
        axs[i,0].set_ylabel(f'{titles[i]}\nCount')
        axs[i,0].set_xlim(*rxbounds)
        axs[i,0].set_ylim(0, rybound)
        axs[i,1].set_xlim(*cxbounds)
        axs[i,1].set_ylim(0, cybound)
    fig.suptitle(variable_name)
    plt.tight_layout()
    if save_name is not None: savefigure(fig, save_name)
    plt.show()

def makeConditionUncertaintyTypePlot(dirs, titles, variable_names, islice=None, jslice=None, figsize=(12,15), save_name=None):
    df = {"variable": [], "condition": [], "uncertainty": [], "type": []}
    for j, d in enumerate(dirs):
        aleatoric_path, epistemic_path = getPaths(d, returns=["aleatoric","epistemic"])
        aleatoric = loadRaster(aleatoric_path, islice=islice, jslice=jslice, elementwise_fn=np.sqrt)
        epistemic = loadRaster(epistemic_path, islice=islice, jslice=jslice, elementwise_fn=np.sqrt)
        mean_aleatoric = np.nanmean(aleatoric, axis=(1,2))
        mean_epistemic = np.nanmean(epistemic, axis=(1,2))
        for i, variable_name in enumerate(variable_names):
            for u, utype in zip([mean_aleatoric[i], mean_epistemic[i]], ["aleatoric", "epistemic"]):
                df["variable"].append(variable_name)
                df["condition"].append(titles[j])
                df["uncertainty"].append(u)
                df["type"].append(utype)
    df = pd.DataFrame(df)
    fig, axs = plt.subplots(nrows=len(variable_names), ncols=2, figsize=figsize)
    for i, variable_name in enumerate(variable_names):
        # average distance to ensemble mean plot
        sub = df.query(f"variable == '{variable_name}'")
        sns.lineplot(
            data=sub,
            x="condition",
            y="uncertainty",
            hue="type",
            ax=axs[i,0]
        )
        axs[i,0].set_ylabel(f"{variable_name}")
        axs[i,0].set_xlabel("")
        if i==0: axs[i,0].set_title(f"average distance to ensemble mean")
        if i==len(variable_names)-1: axs[i,0].legend().set_title("")
        else: axs[i,0].legend([],[],frameon=False)
        # average distance to ensemble mean increase plot
        for utype in ["aleatoric", "epistemic"]:
            sub = df.query(f"variable == '{variable_name}' & type == '{utype}'")
            sub["uncertainty change"] = sub.uncertainty.values-sub.uncertainty.values[0]
            c = "tab:orange" if utype=="epistemic" else "tab:blue"
            axs[i,1].plot(sub.condition, sub["uncertainty change"], color=c, label=utype)
        if i==len(variable_names)-1: axs[i,1].legend()
        if i==0: axs[i,1].set_title(f"average distance to ensemble mean increase")
    plt.tight_layout()
    if save_name is not None: savefigure(fig, save_name)
    plt.show()

def makeConditionUncertaintyCoveragePlot(dirs, titles, variable_names, trainset_means, islice=None, jslice=None, figsize=(12,8), save_name=None):
    df = {"variable": [], "condition": [], "mean predictive uncertainty": [], "mean relative predictive uncertainty (%)": []}
    for j, d in enumerate(dirs):
        variance_path = getPaths(d, returns=["variance"])
        pu = loadRaster(variance_path, islice=islice, jslice=jslice, elementwise_fn=np.sqrt)
        mean_pu = np.nanmean(pu, axis=(1,2))
        for i, variable_name in enumerate(variable_names):
            df["variable"].append(variable_name)
            df["condition"].append(titles[j])
            df["mean predictive uncertainty"].append(mean_pu[i])
            df["mean relative predictive uncertainty (%)"].append(mean_pu[i]/trainset_means[i]*100)
    df = pd.DataFrame(df)
    fig = plt.figure(figsize=figsize)
    sns.lineplot(
        data=df,
        x="condition",
        y="mean relative predictive uncertainty (%)",
        hue="variable"
    )
    plt.tight_layout()
    if save_name is not None: savefigure(fig, save_name)
    plt.show()

def computeExplodedResiduals(result_dirs, gt_dir, s2repr_dirs, variable_names,
                 islice=None, jslice=None, num_bins=15):
    # Compute residual bounds and average cps
    means_bounds = [[np.inf, -np.inf] for _ in variable_names]
    for rd in result_dirs:
        gtp, mp = getPaths(rd, s2repr_dirs=s2repr_dirs, gt_dir=gt_dir, returns=["gt", "mean"])
        gt = loadRaster(gtp, bands=None, islice=islice, jslice=jslice, set_nan_mask=True)
        gt[2] /= 100
        gt[4] /= 100
        mean = loadRaster(mp,bands=None, islice=islice, jslice=jslice)
        mask = np.bitwise_and(~np.isnan(mean).all(0), ~np.isnan(gt).all(0))
        mean = mean[:,mask]
        for i, (rmin, rmax) in enumerate(means_bounds):
            local_min, local_max = np.nanmin(mean[i]), np.nanmax(mean[i])
            if local_min < rmin: means_bounds[i][0] = local_min
            if local_max > rmax: means_bounds[i][1] = local_max
    # Define bins
    bins = np.stack([
        np.linspace(rmin, rmax, num_bins)
        for rmin, rmax in means_bounds
    ], axis=0)
    # Fill bins
    bin_counts = np.zeros((len(result_dirs), len(variable_names), num_bins))
    bin_means = np.zeros((len(result_dirs), len(variable_names), num_bins))
    bin_gts = np.zeros((len(result_dirs), len(variable_names), num_bins))
    for k, rd in enumerate(result_dirs):
        # load
        gtp, mp = getPaths(rd, s2repr_dirs=s2repr_dirs, gt_dir=gt_dir, returns=["gt", "mean"])
        gt = loadRaster(gtp, bands=None, islice=islice, jslice=jslice, set_nan_mask=True)
        gt[2] /= 100
        gt[4] /= 100
        mean = loadRaster(mp,bands=None, islice=islice, jslice=jslice)
        mask = np.bitwise_and(~np.isnan(mean).all(0), ~np.isnan(gt).all(0))
        gt = gt[:,mask]
        mean = mean[:,mask]
        # digitize
        bin_ids = np.stack(
            [np.digitize(mean[i], bins=bins[i])-1 for i in range(len(variable_names))],
            axis=0            
        )
        # insert in bins: running average
        for bin_id in range(num_bins):
            mask = (bin_ids==bin_id)
            for i, variable_mask in enumerate(mask):
                count = mask[i].sum()
                if count==0: continue
                bin_means[k,i,bin_id] = (bin_counts[k,i,bin_id]*bin_means[k,i,bin_id] + np.nansum(mean[i,variable_mask])) / (bin_counts[k,i,bin_id]+count)
                bin_gts[k,i,bin_id] = (bin_counts[k,i,bin_id]*bin_gts[k,i,bin_id] + np.nansum(gt[i,variable_mask])) / (bin_counts[k,i,bin_id]+count)
                bin_counts[k,i,bin_id] += count
    return bin_means, bin_gts, bin_counts

def explodedRMSEplot(bin_means, bin_gts, variable_names, variable_indexes, avg_cps, figsize=(12,6), ncols=2, save_name=None, hi_bound=np.inf):
    if not isinstance(hi_bound, list): hi_bound = [hi_bound for _ in variable_names]
    nrows = max(1, len(variable_names)//ncols)
    fig, axs = plt.subplots(ncols=ncols,nrows=nrows,figsize=figsize)
    if len(variable_names)>1: axs = axs.flatten()
    else: axs = [axs]
    assert len(variable_names)==len(variable_indexes)
    num_bins = bin_means.shape[-1]
    for i, (varname, varindex) in enumerate(zip(variable_names, variable_indexes)):
        # select variable
        bmean, bgt = bin_means[:,varindex], bin_gts[:,varindex] # num_images x num_bins
        # construct dataframe
        data = {
            "average cloud probability": [],
            "binned ground truth": [],
            "binned mean": [],
        }
        for k in range(len(avg_cps)): # loop over images to fill data
            data["average cloud probability"].extend([avg_cps[k] for _ in range(num_bins)])
            bgtk, bmeank = zip(*sorted(zip(bgt[k],bmean[k]), key=lambda v:v[0]))
            data["binned ground truth"].extend(list(bgtk))
            data["binned mean"].extend(list(bmeank))
        data = pd.DataFrame(data)
        data = data.sort_values(by="average cloud probability")
        data = data[data["binned ground truth"]<hi_bound[i]]
        lo = min(data["binned ground truth"].min(), data["binned mean"].min())
        hi = min(data["binned ground truth"].max(), data["binned mean"].max())
        id_line = (
            np.linspace(lo, hi, 2), 
            np.linspace(lo, hi, 2)
        )
        axs[i].plot(*id_line, color="black", linestyle="dotted")
        sns.lineplot(data, x="binned ground truth", y="binned mean", palette="bwr", hue="average cloud probability", ax=axs[i], alpha=0.3)
        axs[i].legend().remove()
        if len(variable_names)>1: axs[i].set_title(varname)
        cmap = sns.color_palette("bwr")
        b, r = cmap[0], cmap[-1]
        kmax = np.argmax(avg_cps)
        bgtk, bmeank = zip(*sorted(zip(bgt[kmax],bmean[kmax]), key=lambda v:v[0]))
        bgtk, bmeank = np.array(bgtk), np.array(bmeank)
        mask = bgtk<hi_bound[i]
        bgtk, bmeank = bgtk[mask], bmeank[mask]
        axs[i].plot(bgtk, bmeank, color=r, label=f"maximum: {avg_cps[kmax]:.2f}%")
        kmin = np.argmin(avg_cps)
        bgtk, bmeank = zip(*sorted(zip(bgt[kmin],bmean[kmin]), key=lambda v:v[0]))
        bgtk, bmeank = np.array(bgtk), np.array(bmeank)
        mask = bgtk<hi_bound[i]
        bgtk, bmeank = bgtk[mask], bmeank[mask]        
        axs[i].plot(bgtk, bmeank, color=b, label=f"minimum: {avg_cps[kmin]:.2f}%")
        handles, labels = axs[i].get_legend_handles_labels()
        axs[i].legend([handle for i,handle in enumerate(handles[-2:])],
                    [label for i,label in enumerate(labels[-2:])],
                    title="average cloud probability",
                    loc= 'best')
    sm = plt.cm.ScalarMappable(cmap="bwr", norm=plt.Normalize(min(avg_cps), max(avg_cps)))
    axs[-1].figure.colorbar(sm, ax=axs[-1]) # only last one here
    plt.tight_layout()
    if save_name is not None: savefigure(fig, save_name)
    plt.show()

def compareS2Bands(matching_dir, s2repr_dirs_original, s2repr_dirs_gee):
    def getDataPositions(*xs, nodata=0.):
        valid_pos = [[] for _ in range(len(xs))]
        for i in range(xs[0].shape[1]):
            for j in range(xs[0].shape[2]):
                for x, vp in zip(xs, valid_pos):
                    if not (x[:,i,j]==nodata).all(): vp.append((i,j))
        valid_pos = [list(zip(*x)) for x in valid_pos]
        return tuple(valid_pos)
    df = {"band": list(range(1,13)), "min": [], "max": [], "mean": [], "median": []}
    od, gd = matching_dir
    os2 = loadRaster(getPaths(od, s2repr_dirs=s2repr_dirs_original, returns=["img"]), dtype=float)[:12]
    gs2 = loadRaster(getPaths(gd, s2repr_dirs=s2repr_dirs_gee, returns=["img"]), dtype=float)[:12]
    assert os2.shape==gs2.shape
    # valid_locs = np.bitwise_or((os2==0.).all(0),(gs2==0.).all(0))
    # os2 = os2[:,valid_locs]
    # gs2 = gs2[:,valid_locs]
    opos, gpos = getDataPositions(os2, gs2, nodata=0.)
    os2 = os2[:,opos[0],opos[1]].reshape(12,-1)
    gs2 = gs2[:,gpos[0],gpos[1]].reshape(12,-1)
    for f, fn in zip([np.nanmin, np.nanmax, np.nanmean, np.nanmedian], 
                        ["min", "max", "mean", "median"]):
        of, gf = f(os2, axis=1), f(gs2, axis=1)
        of2 = of.copy()
        of2[of==gf] = 1. # avoid division by 0 when 0-0
        rdelta = (gf-of)/of2*100
        df[fn] = rdelta 
    return pd.DataFrame(df)
    
def CompareMapsStats(matching_dir, variable_names, split_mask=None, split=None):
    mean_stats = {"variable": [], "stat": [], "original": [], "gee": [], "delta": [], "relative delta (%)": []}
    pu_stats = {"variable": [], "stat": [], "original": [], "gee": [], "delta": [], "relative delta (%)": []}
    if split_mask is not None or split is not None: 
        assert split_mask is not None and split is not None
        split_ = split
        split = ["train", "val", "test"].index(split)
        mean_stats["split"] = []
        pu_stats["split"] = []
    (orig_dir, gee_dir) = matching_dir    
    #for i, (orig_dir, gee_dir) in enumerate(matching_dirs):
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

def absoluteToRelativeScale(value, reference):
    return value / reference

def relativeToAbsoluteScale(value, reference):
    return value * reference

def plotTrainTestMetricsDataFrames(train_df, test_df, type="bar", save_name=None, 
                                   variables=None, metrics=None, split_splits=True, 
                                   add_relative=True, figsize=(15,10)):
    assert type in ["bar", "scatter"]
    metrics_df = pd.concat([train_df, test_df])
    variables = variables if variables is not None else metrics_df.variable.unique()
    metrics = metrics if metrics is not None else metrics_df.metric.unique()
    fig, axs = plt.subplots(
        ncols=len(metrics),
        nrows=len(variables),
        figsize=figsize
    )
    for i, v in enumerate(variables):
        for j, m in enumerate(metrics):
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
                tmp = tmp.groupby(by=["source", "split"]).mean().reset_index()
                kw = {"hue_order": ["original", "gee"]} if split_splits else {}
                sns.barplot(data=tmp, x=xx, y="x", hue=hue, ax=axs[i,j], 
                            errorbar=None, order=["train", "test"], **kw)
                if add_relative:
                    train_reference = tmp.query("source == 'original' & split == 'train'").x.values[0]
                    tmp2 = tmp.copy()
                    tmp2.x = absoluteToRelativeScale(tmp2.x, train_reference)
                    sec_ax = axs[i,j].twinx()
                    sns.barplot(data=tmp2, x=xx, y="x", hue=hue, ax=sec_ax, 
                                errorbar=None, order=["train", "test"], **kw)
                    sec_ax.grid(False)
            try: 
                if i==len(variables)-1 and j == len(metrics)-1:
                    axs[i,j].legend().set_title("")
                    sec_ax.get_legend().remove()
                else:
                    for a in [axs[i,j], sec_ax]:
                        a.get_legend().remove()                    
            except: continue
            for a in [axs[i,j], sec_ax]:
                a.set_xlabel("")
                a.set_ylabel("")
            if i==0:
                axs[i,j].set_title(m)
            if j==0:
                axs[i,j].set_ylabel(f"{v}\n(absolute)")
            if j==len(metrics)-1:
                sec_ax.set_ylabel(f"{v}\n(relative)")
            if i==len(variables)-1 and j == len(metrics)-1: axs
    plt.tight_layout()
    # fig = single_legend_figure(fig, 2)
    if save_name is not None: savefigure(fig, save_name)
    plt.show()

def singlePlotsCloudVsUncertaintyType(
    dirs,
    s2repr_dirs,
    variable_indexes,
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
        aleatoric = loadRaster(aleatoric_path, bands=variable_indexes, dtype="float64", elementwise_fn=np.sqrt).reshape(nv, -1)
        epistemic = loadRaster(epistemic_path, bands=variable_indexes, dtype="float64", elementwise_fn=np.sqrt).reshape(nv, -1)
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

    # Absolute plot
    fig, axs = plt.subplots(ncols=1, nrows=nv, figsize=figsize)
    for i, variable_name in enumerate(variable_names):
        ax = axs if nv==1 else axs[i]
        sub = df.query(f"variable == '{variable_name}'")
        sns.lineplot(data=sub, x="cp", y="average uncertainty", hue="type", ax=ax)
        ax.fill_between(sub[sub.type=="aleatoric"].cp, 
                        sub[sub.type=="aleatoric"]["average uncertainty"]-sub[sub.type=="aleatoric"]["uncertainty std"], 
                        sub[sub.type=="aleatoric"]["average uncertainty"]+sub[sub.type=="aleatoric"]["uncertainty std"], 
                        color="b",
                        alpha=0.2)
        ax.fill_between(sub[sub.type=="epistemic"].cp, 
                        sub[sub.type=="epistemic"]["average uncertainty"]-sub[sub.type=="epistemic"]["uncertainty std"], 
                        sub[sub.type=="epistemic"]["average uncertainty"]+sub[sub.type=="epistemic"]["uncertainty std"], 
                        color="orange",
                        alpha=0.2)
        ax.set_ylabel(f"average uncertainty")
        ax.set_xlabel(f"cloud probability")
        ax.legend([],[],frameon=False)
    plt.tight_layout()
    if save_name is not None: savefigure(fig, save_name+"-absolute")
    plt.show()

    # Aleatoric plot
    fig, axs = plt.subplots(ncols=1, nrows=nv, figsize=figsize)
    for i, variable_name in enumerate(variable_names):
        ax = axs if nv==1 else axs[i]
        sub = df.query(f"variable == '{variable_name}' & type == 'aleatoric'")
        sns.lineplot(data=sub, x="cp", y="average uncertainty", ax=ax)
        ax.set_ylabel(f"average ensemble aleatoric uncertainty")
        ax.set_xlabel(f"cloud probability")
        ax.legend([],[],frameon=False)
    plt.tight_layout()
    if save_name is not None: savefigure(fig, save_name+"-aleatoric")
    plt.show() 

    # Epistemic plot
    fig, axs = plt.subplots(ncols=1, nrows=nv, figsize=figsize)
    for i, variable_name in enumerate(variable_names):
        ax = axs if nv==1 else axs[i]
        sub = df.query(f"variable == '{variable_name}' & type == 'epistemic'")
        sub["average uncertainty"] = sub["average uncertainty"]**2
        sns.lineplot(data=sub, x="cp", y="average uncertainty", ax=ax)
        ax.set_ylabel(f"average ensemble squared epistemic uncertainty")
        ax.set_xlabel(f"cloud probability")
        ax.legend([],[],frameon=False)
    plt.tight_layout()
    if save_name is not None: savefigure(fig, save_name+"-epistemic")
    plt.show() 

    # fraction plot
    fig, axs = plt.subplots(ncols=1, nrows=nv, figsize=figsize)
    df["uncertainty_fraction"] = [None for _ in range(df.shape[0])]
    total_avg_uncertainty = df.groupby(["variable", "cp"]).sum()["average uncertainty"].reset_index()
    for i, variable_name in enumerate(variable_names):
        ax = axs if nv==1 else axs[i]
        for utype in ["aleatoric", "epistemic"]:
            values = df.query(f"variable == '{variable_name}' & type == '{utype}'").sort_values("cp")
            fracs = values["average uncertainty"].values / total_avg_uncertainty.query(f"variable == '{variable_name}'").sort_values("cp")["average uncertainty"].values
            c = "tab:orange" if utype=="epistemic" else "tab:blue"
            ax.plot(values.cp, fracs, color=c, label=utype)
        ax.hlines(0.5, xmin=0, xmax=100, linestyle="dotted", color="black")
        ax.set_ylabel(f"average uncertainty fraction")
        ax.set_xlabel(f"cloud probability")
        ax.legend([],[],frameon=False)
    plt.tight_layout()
    if save_name is not None: savefigure(fig, save_name+"-fraction")
    plt.show()

    # relative plot
    fig, axs = plt.subplots(ncols=1, nrows=nv, figsize=figsize)
    for i, variable_name in enumerate(variable_names):
        ax = axs if nv==1 else axs[i]
        for utype in ["aleatoric", "epistemic"]:
            sub = df.query(f"variable == '{variable_name}' & type == '{utype}'")
            sub["uncertainty change"] = sub["average uncertainty"].values-sub["average uncertainty"].values[0]
            c = "tab:orange" if utype=="epistemic" else "tab:blue"
            ax.plot(sub.cp, sub["uncertainty change"], color=c, label=utype)
        ax.set_ylabel(f"relative average uncertainty")
        ax.set_xlabel(f"cloud probability")
        if i==nv-1: ax.legend()
    plt.tight_layout()
    if save_name is not None: savefigure(fig, save_name+"-relative")
    plt.show()

def singlePlotsCloudVsMembersUncertainty(
    dirs,
    s2repr_dirs,
    variable_indexes,
    variable_names,
    figsize=(12,18),
    save_name=None
):
    nv = len(variable_names)
    df = pd.DataFrame(
        columns=["cp", "count", "average uncertainty", "uncertainty std", "model", "variable"])
    # loop on directories
    for i, d in enumerate(dirs):
        # read data
        variances_path, img_path, aleatoric_path = getPaths(
            d, s2repr_dirs=s2repr_dirs, returns=["models_variance", "img", "aleatoric"]
        )
        variances = loadRaster(variances_path, dtype="float64", elementwise_fn=np.sqrt).reshape(5, 5, -1).transpose(1,0,2) # (M, Var, samples) => (Var, M, Samples)
        variances = variances[np.array(variable_indexes)-1].reshape(nv, 5, -1) # (Var, M, Samples)
        aleatoric = loadRaster(aleatoric_path, bands=variable_indexes, dtype="float64", elementwise_fn=np.sqrt).reshape(nv, -1)
        cp = loadRaster(img_path, bands=-1, dtype="float").reshape(-1)
        mask = ~np.isnan(variances).all((0, 1))
        variances, cp, aleatoric = variances[:,:,mask], cp[mask], aleatoric[:,mask]
        tmp = {
            "cp": [],
            "count": [],
            "average uncertainty": [],
            "uncertainty std": [],
            "model": [],
            "variable": [],
        }
        for i, variable_name in enumerate(variable_names):
            for j, array in enumerate(variances[i]):
                tmp["cp"].extend(cp.tolist())
                tmp["count"].extend(np.ones_like(cp).tolist())
                tmp[f"average uncertainty"].extend(array.tolist())
                tmp[f"uncertainty std"].extend((array**2).tolist())
                tmp[f"model"].extend([str(j+1) for _ in array])
                tmp[f"variable"].extend([variable_name for _ in array])
            tmp["cp"].extend(cp.tolist())
            tmp["count"].extend(np.ones_like(cp).tolist())
            tmp[f"average uncertainty"].extend(aleatoric[i].tolist())
            tmp[f"uncertainty std"].extend((aleatoric[i]**2).tolist())
            tmp[f"model"].extend(["ensemble" for _ in aleatoric[i]])
            tmp[f"variable"].extend([variable_name for _ in aleatoric[i]])
        tmp = pd.DataFrame(tmp)
        tmp = tmp.groupby(["cp", "model", "variable"]).sum().reset_index()
        df = pd.concat([df, tmp], axis=0)
    df = df.groupby(["cp", "variable", "model"]).sum().reset_index()
    df["average uncertainty"] /= df["count"]
    df["uncertainty std"] /= df["count"]
    df["uncertainty std"] = np.sqrt(df["uncertainty std"]-df["average uncertainty"]**2)

    # Absolute
    fig, axs = plt.subplots(ncols=1, nrows=nv, figsize=figsize)
    colors = sns.color_palette()
    for i, variable_name in enumerate(variable_names):
        ax = axs if nv==1 else axs[i]
        sub = df.query(f"variable == '{variable_name}'")
        sns.lineplot(data=sub, x="cp", y="average uncertainty", hue="model", hue_order=[str(j) for j in range(1,6)], ax=ax)
        for j in range(1,6):
            subsub = sub.query(f"model == '{j}'")
            ax.fill_between(subsub.cp, 
                                  subsub["average uncertainty"]-subsub["uncertainty std"], 
                                  subsub["average uncertainty"]+subsub["uncertainty std"], 
                                  color=colors[j-1],
                                  alpha=0.1)
        ax.set_xlabel(f"cloud probability")
        ax.set_ylabel(f"average aleatoric uncertainty")
        ax.legend([],[],frameon=False)
    plt.tight_layout()
    if save_name is not None: savefigure(fig, save_name+"-absolute")
    plt.show()

    # Absolute with ensemble
    fig, axs = plt.subplots(ncols=1, nrows=nv, figsize=figsize)
    colors = sns.color_palette()
    for i, variable_name in enumerate(variable_names):
        ax = axs if nv==1 else axs[i]
        sub = df.query(f"variable == '{variable_name}'")
        sns.lineplot(data=sub, x="cp", y="average uncertainty", hue="model", hue_order=[str(j) for j in range(1,6)], ax=ax)
        for j in range(1,6):
            subsub = sub.query(f"model == '{j}'")
            ax.fill_between(subsub.cp, 
                                  subsub["average uncertainty"]-subsub["uncertainty std"], 
                                  subsub["average uncertainty"]+subsub["uncertainty std"], 
                                  color=colors[j-1],
                                  alpha=0.1)
        # ensemble plot
        sub = df.query(f"variable == '{variable_name}' & model == 'ensemble'")
        ax.plot(sub.cp, sub["average uncertainty"], linestyle="dotted", color="k")
        ax.set_xlabel(f"cloud probability")
        ax.set_ylabel(f"average aleatoric uncertainty")
        ax.legend([],[],frameon=False)
    plt.tight_layout()
    if save_name is not None: savefigure(fig, save_name+"-absolute_w_ensemble")
    plt.show()

    # Relative
    fig, axs = plt.subplots(ncols=1, nrows=nv, figsize=figsize)
    colors = sns.color_palette()
    for i, variable_name in enumerate(variable_names):
        ax = axs if nv==1 else axs[i]
        for j in range(1,6):
            sub = df.query(f"variable == '{variable_name}' & model == '{j}'")
            sub["uncertainty change"] = sub["average uncertainty"].values-sub["average uncertainty"].values[0]
            ax.plot(sub.cp, sub["uncertainty change"], color=colors[j-1], label=str(j))
        if i==nv-1: ax.legend(title="model", loc="center left", bbox_to_anchor=(1,0.5))
        ax.set_xlabel(f"cloud probability")
        ax.set_ylabel(f"relative average aleatoric uncertainty")
    plt.tight_layout()
    if save_name is not None: savefigure(fig, save_name+"-relative")
    plt.show()

    # Relative with ensemble
    fig, axs = plt.subplots(ncols=1, nrows=nv, figsize=figsize)
    colors = sns.color_palette()
    for i, variable_name in enumerate(variable_names):
        ax = axs if nv==1 else axs[i]
        for j in list(range(1,6))+["ensemble"]:
            kwargs = dict(color=colors[j-1], label=str(j)) if j in range(1,6) else \
                     dict(linestyle="dotted", color="k", label="Ens")
            sub = df.query(f"variable == '{variable_name}' & model == '{j}'")
            sub["uncertainty change"] = sub["average uncertainty"].values-sub["average uncertainty"].values[0]
            ax.plot(sub.cp, sub["uncertainty change"], **kwargs)
        if i==nv-1: ax.legend(title="model", loc="center left", bbox_to_anchor=(1,0.5))
        ax.set_xlabel(f"cloud probability")
        ax.set_ylabel(f"relative average aleatoric uncertainty")
    plt.tight_layout()
    if save_name is not None: savefigure(fig, save_name+"-relative_w_ensemble")
    plt.show()

def singlePlotsCloudVsMembersEpistemic(
    dirs,
    s2repr_dirs,
    variable_indexes,
    variable_names,
    figsize=(12,18),
    save_name=None
):
    nv = len(variable_names)
    df = pd.DataFrame(
        columns=["cp", "count", "average squared epistemic uncertainty", "squared epistemic uncertainty std", "model", "variable"])
    # loop on directories
    for i, d in enumerate(dirs):
        # read data
        mean_path, img_path, epistemic_path = getPaths(
            d, s2repr_dirs=s2repr_dirs, returns=["models_mean", "img", "epistemic"]
        )
        means = loadRaster(mean_path, dtype="float64").reshape(5, 5, -1).transpose(1,0,2) # (M, allVar, samples) => (allVar, M, Samples)
        means = means[np.array(variable_indexes)-1].reshape(nv, 5, -1) # (Var, M, Samples)
        epistemic = loadRaster(epistemic_path, bands=variable_indexes, dtype="float64").reshape(nv, -1)
        cp = loadRaster(img_path, bands=-1, dtype="float").reshape(-1)
        mask = ~np.isnan(means).all((0, 1))
        means, cp, epistemic = means[:,:,mask], cp[mask], epistemic[:,mask]
        tmp = {
            "cp": [],
            "count": [],
            "average squared epistemic uncertainty": [],
            "squared epistemic uncertainty std": [],
            "model": [],
            "variable": [],
        }
        mean_means = np.nanmean(means, axis=1)
        for i, variable_name in enumerate(variable_names):
            for j, array in enumerate(means[i]):
                array = (means[i,j]-mean_means[i])**2
                tmp["cp"].extend(cp.tolist())
                tmp["count"].extend(np.ones_like(cp).tolist())
                tmp[f"average squared epistemic uncertainty"].extend(array.tolist())
                tmp[f"squared epistemic uncertainty std"].extend((array**2).tolist())
                tmp[f"model"].extend([str(j+1) for _ in array])
                tmp[f"variable"].extend([variable_name for _ in array])
            tmp["cp"].extend(cp.tolist())
            tmp["count"].extend(np.ones_like(cp).tolist())
            tmp[f"average squared epistemic uncertainty"].extend(epistemic[i].tolist())
            tmp[f"squared epistemic uncertainty std"].extend((epistemic[i]**2).tolist())
            tmp[f"model"].extend(["ensemble" for _ in epistemic[i]])
            tmp[f"variable"].extend([variable_name for _ in epistemic[i]])
        tmp = pd.DataFrame(tmp)
        tmp = tmp.groupby(["cp", "model", "variable"]).sum().reset_index()
        df = pd.concat([df, tmp], axis=0)
    df = df.groupby(["cp", "variable", "model"]).sum().reset_index()
    df["average squared epistemic uncertainty"] /= df["count"]
    df["squared epistemic uncertainty std"] /= df["count"]
    df["squared epistemic uncertainty std"] = np.sqrt(df["squared epistemic uncertainty std"]-df["average squared epistemic uncertainty"]**2)
    
    # Absolute
    fig, axs = plt.subplots(ncols=1, nrows=nv, figsize=figsize)
    colors = sns.color_palette()
    for i, variable_name in enumerate(variable_names):
        ax = axs if nv==1 else axs[i]
        sub = df.query(f"variable == '{variable_name}' & model != 'ensemble'")
        sns.lineplot(data=sub, x="cp", y="average squared epistemic uncertainty", hue="model", hue_order=[str(j) for j in range(1,6)], ax=ax)
        for j in range(1,6):
            subsub = sub.query(f"model == '{j}'")
            ax.fill_between(subsub.cp, 
                                  subsub["average squared epistemic uncertainty"]-subsub["squared epistemic uncertainty std"], 
                                  subsub["average squared epistemic uncertainty"]+subsub["squared epistemic uncertainty std"], 
                                  color=colors[j-1],
                                  alpha=0.1)
        ax.set_ylabel(f"average squared epistemic uncertainty")
        ax.set_xlabel(f"cloud probability")
        # if i==0: ax.set_title(f"average squared epistemic uncertainty")
        ax.legend([],[],frameon=False)
    plt.tight_layout()
    if save_name is not None: savefigure(fig, save_name+"-absolute")
    plt.show()

    # Absolute with ensemble
    fig, axs = plt.subplots(ncols=1, nrows=nv, figsize=figsize)
    colors = sns.color_palette()
    for i, variable_name in enumerate(variable_names):
        ax = axs if nv==1 else axs[i]
        sub = df.query(f"variable == '{variable_name}' & model != 'ensemble'")
        sns.lineplot(data=sub, x="cp", y="average squared epistemic uncertainty", hue="model", hue_order=[str(j) for j in range(1,6)], ax=ax)
        for j in range(1,6):
            subsub = sub.query(f"model == '{j}'")
            ax.fill_between(subsub.cp, 
                                  subsub["average squared epistemic uncertainty"]-subsub["squared epistemic uncertainty std"], 
                                  subsub["average squared epistemic uncertainty"]+subsub["squared epistemic uncertainty std"], 
                                  color=colors[j-1],
                                  alpha=0.1)
        # ensemble plot
        sub = df.query(f"variable == '{variable_name}' & model == 'ensemble'")
        ax.plot(sub.cp, sub["average squared epistemic uncertainty"], linestyle="dotted", color="k")
        ax.set_ylabel(f"average squared epistemic uncertainty")
        ax.set_xlabel(f"cloud probability")
        ax.legend([],[],frameon=False)
    plt.tight_layout()
    if save_name is not None: savefigure(fig, save_name+"-absolute_w_ensemble")
    plt.show()

    # Relative
    fig, axs = plt.subplots(ncols=1, nrows=nv, figsize=figsize)    
    for i, variable_name in enumerate(variable_names):
        ax = axs if nv==1 else axs[i]
        for j in range(1,6):
            sub = df.query(f"variable == '{variable_name}' & model == '{j}'")
            sub["uncertainty change"] = sub["average squared epistemic uncertainty"].values-sub["average squared epistemic uncertainty"].values[0]
            ax.plot(sub.cp, sub["uncertainty change"], color=colors[j-1], label=str(j))
        if i==nv-1: ax.legend(title="model", loc="center left", bbox_to_anchor=(1,0.5))
        ax.set_ylabel(f"relative average squared epistemic uncertainty")
        ax.set_xlabel(f"cloud probability")
    plt.tight_layout()
    if save_name is not None: savefigure(fig, save_name+"-relative")
    plt.show()

    # Relative with ensemble
    fig, axs = plt.subplots(ncols=1, nrows=nv, figsize=figsize)    
    for i, variable_name in enumerate(variable_names):
        ax = axs if nv==1 else axs[i]
        for j in list(range(1,6))+["ensemble"]:
            kwargs = dict(color=colors[j-1], label=str(j)) if j in range(1,6) else \
                     dict(linestyle="dotted", color="k", label="Ens")
            sub = df.query(f"variable == '{variable_name}' & model == '{j}'")
            sub["uncertainty change"] = sub["average squared epistemic uncertainty"].values-sub["average squared epistemic uncertainty"].values[0]
            ax.plot(sub.cp, sub["uncertainty change"], **kwargs)
        if i==nv-1: ax.legend(title="model", loc="center left", bbox_to_anchor=(1,0.5))
        ax.set_ylabel(f"relative average squared epistemic uncertainty")
        ax.set_xlabel(f"cloud probability")
    plt.tight_layout()
    if save_name is not None: savefigure(fig, save_name+"-relative_w_ensemble")
    plt.show()

def showCloudVsMembersUncertainty(
    dirs,
    s2repr_dirs,
    variable_indexes,
    variable_names,
    figsize=(12,18),
    save_name=None
):
    nv = len(variable_names)
    df = pd.DataFrame(
        columns=["cp", "count", "average uncertainty", "uncertainty std", "model", "variable"])
    # loop on directories
    for i, d in enumerate(dirs):
        # read data
        variances_path, img_path = getPaths(
            d, s2repr_dirs=s2repr_dirs, returns=["models_variance", "img"]
        )
        variances = loadRaster(variances_path, dtype="float64", elementwise_fn=np.sqrt).reshape(5, 5, -1).transpose(1,0,2) # (M, Var, samples) => (Var, M, Samples)
        variances = variances[np.array(variable_indexes)-1].reshape(nv, 5, -1) # (Var, M, Samples)
        cp = loadRaster(img_path, bands=-1, dtype="float").reshape(-1)
        mask = ~np.isnan(variances).all((0, 1))
        variances, cp = variances[:,:,mask], cp[mask]
        tmp = {
            "cp": [],
            "count": [],
            "average uncertainty": [],
            "uncertainty std": [],
            "model": [],
            "variable": [],
        }
        for i, variable_name in enumerate(variable_names):
            for j, array in enumerate(variances[i]):
                tmp["cp"].extend(cp.tolist())
                tmp["count"].extend(np.ones_like(cp).tolist())
                tmp[f"average uncertainty"].extend(array.tolist())
                tmp[f"uncertainty std"].extend((array**2).tolist())
                tmp[f"model"].extend([str(j+1) for _ in array])
                tmp[f"variable"].extend([variable_name for _ in array])
        tmp = pd.DataFrame(tmp)
        tmp = tmp.groupby(["cp", "model", "variable"]).sum().reset_index()
        df = pd.concat([df, tmp], axis=0)
    df = df.groupby(["cp", "variable", "model"]).sum().reset_index()
    df["average uncertainty"] /= df["count"]
    df["uncertainty std"] /= df["count"]
    df["uncertainty std"] = np.sqrt(df["uncertainty std"]-df["average uncertainty"]**2)
    fig, axs = plt.subplots(ncols=2, nrows=nv, figsize=figsize)
    colors = sns.color_palette()
    for i, variable_name in enumerate(variable_names):
        ax = axs[0] if nv==1 else axs[i,0]
        sub = df.query(f"variable == '{variable_name}'")
        sns.lineplot(data=sub, x="cp", y="average uncertainty", hue="model", hue_order=[str(j) for j in range(1,6)], ax=ax)
        for j in range(1,6):
            subsub = sub.query(f"model == '{j}'")
            ax.fill_between(subsub.cp, 
                                  subsub["average uncertainty"]-subsub["uncertainty std"], 
                                  subsub["average uncertainty"]+subsub["uncertainty std"], 
                                  color=colors[j-1],
                                  alpha=0.1)
        ax.set_ylabel(f"{variable_name}")
        ax.set_xlabel(f"cloud probability")
        if i==0: ax.set_title(f"average aleatoric uncertainty")
        ax.legend([],[],frameon=False)
    for i, variable_name in enumerate(variable_names):
        ax = axs[1] if nv==1 else axs[i,1]
        for j in range(1,6):
            sub = df.query(f"variable == '{variable_name}' & model == '{j}'")
            sub["uncertainty change"] = sub["average uncertainty"].values-sub["average uncertainty"].values[0]
            ax.plot(sub.cp, sub["uncertainty change"], color=colors[j-1], label=str(j))
        ax.set_xlabel(f"cloud probability")
        if i==nv-1: ax.legend(title="model", loc="center left", bbox_to_anchor=(1,0.5))
        if i==0: ax.set_title(f"relative average aleatoric uncertainty")
    plt.tight_layout()
    if save_name is not None: savefigure(fig, save_name)
    plt.show()

def showCloudVsMembersEpistemic(
    dirs,
    s2repr_dirs,
    variable_indexes,
    variable_names,
    figsize=(12,18),
    save_name=None
):
    nv = len(variable_names)
    df = pd.DataFrame(
        columns=["cp", "count", "average squared epistemic uncertainty", "squared epistemic uncertainty std", "model", "variable"])
    # loop on directories
    for i, d in enumerate(dirs):
        # read data
        mean_path, img_path = getPaths(
            d, s2repr_dirs=s2repr_dirs, returns=["models_mean", "img"]
        )
        means = loadRaster(mean_path, dtype="float64").reshape(5, 5, -1).transpose(1,0,2) # (M, allVar, samples) => (allVar, M, Samples)
        means = means[np.array(variable_indexes)-1].reshape(nv, 5, -1) # (Var, M, Samples)
        cp = loadRaster(img_path, bands=-1, dtype="float").reshape(-1)
        mask = ~np.isnan(means).all((0, 1))
        means, cp = means[:,:,mask], cp[mask]
        tmp = {
            "cp": [],
            "count": [],
            "average squared epistemic uncertainty": [],
            "squared epistemic uncertainty std": [],
            "model": [],
            "variable": [],
        }
        mean_means = np.nanmean(means, axis=1)
        for i, variable_name in enumerate(variable_names):
            for j, array in enumerate(means[i]):
                array = (means[i,j]-mean_means[i])**2
                tmp["cp"].extend(cp.tolist())
                tmp["count"].extend(np.ones_like(cp).tolist())
                tmp[f"average squared epistemic uncertainty"].extend(array.tolist())
                tmp[f"squared epistemic uncertainty std"].extend((array**2).tolist())
                tmp[f"model"].extend([str(j+1) for _ in array])
                tmp[f"variable"].extend([variable_name for _ in array])
        tmp = pd.DataFrame(tmp)
        tmp = tmp.groupby(["cp", "model", "variable"]).sum().reset_index()
        df = pd.concat([df, tmp], axis=0)
    df = df.groupby(["cp", "variable", "model"]).sum().reset_index()
    df["average squared epistemic uncertainty"] /= df["count"]
    df["squared epistemic uncertainty std"] /= df["count"]
    df["squared epistemic uncertainty std"] = np.sqrt(df["squared epistemic uncertainty std"]-df["average squared epistemic uncertainty"]**2)
    fig, axs = plt.subplots(ncols=2, nrows=nv, figsize=figsize)
    colors = sns.color_palette()
    for i, variable_name in enumerate(variable_names):
        ax = axs[0] if nv==1 else axs[i,0]
        sub = df.query(f"variable == '{variable_name}'")
        sns.lineplot(data=sub, x="cp", y="average squared epistemic uncertainty", hue="model", hue_order=[str(j) for j in range(1,6)], ax=ax)
        for j in range(1,6):
            subsub = sub.query(f"model == '{j}'")
            ax.fill_between(subsub.cp, 
                                  subsub["average squared epistemic uncertainty"]-subsub["squared epistemic uncertainty std"], 
                                  subsub["average squared epistemic uncertainty"]+subsub["squared epistemic uncertainty std"], 
                                  color=colors[j-1],
                                  alpha=0.1)
        ax.set_ylabel(f"{variable_name}")
        ax.set_xlabel(f"cloud probability")
        if i==0: ax.set_title(f"average squared epistemic uncertainty")
        ax.legend([],[],frameon=False)
    for i, variable_name in enumerate(variable_names):
        ax = axs[1] if nv==1 else axs[i,1]
        for j in range(1,6):
            sub = df.query(f"variable == '{variable_name}' & model == '{j}'")
            sub["uncertainty change"] = sub["average squared epistemic uncertainty"].values-sub["average squared epistemic uncertainty"].values[0]
            ax.plot(sub.cp, sub["uncertainty change"], color=colors[j-1], label=str(j))
        ax.set_xlabel(f"cloud probability")
        if i==nv-1: ax.legend(title="model", loc="center left", bbox_to_anchor=(1,0.5))
        if i==0: ax.set_title(f"relative average squared epistemic uncertainty")
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

def residualNormalityAnalysis(dirs, gt_dir, variable_name, variable_index, exp_vars, figsize=(4,12), save_name=None):
    q = np.linspace(5e-4, 1-5e-4, 400)
    bins = np.linspace(-5,5,400)
    # load gt
    gtp = getPaths(dirs[0], gt_dir=gt_dir, returns=["gt"])
    gt = loadRaster(gtp, bands=variable_index, set_nan_mask=True).reshape(-1)
    if variable_index in [3,5]: gt /=100
    gt_mask = ~np.isnan(gt)
    # normal distribution
    dist = scipy.stats.norm(loc=0, scale=1)
    norm_qs = [dist.ppf(qq) for qq in q]
    # load data
    df_dens = {"average cloud probability": [], "density": [], "bins": []}
    max_res_dens = None
    min_res_dens = None
    df_q = {"average cloud probability": [], "empirical quantiles": [], "theoretical quantiles": []}
    min_res_qs = None
    max_res_qs = None
    for i, d in enumerate(dirs):
        mean = loadRaster(getPaths(d, returns=["mean"]), bands=variable_index).reshape(-1)
        mask = np.bitwise_and(gt_mask, ~np.isnan(mean))
        # standardized residuals
        res = mean[mask]-gt[mask]
        res = (res-res.mean())/res.std()
        res_qs = [np.quantile(res, qq) for qq in q]
        # density
        dens, _ = np.histogram(res, bins, density=True)
        # store
        df_q["average cloud probability"].extend([exp_vars[i] for _ in q])
        df_q["empirical quantiles"].extend(res_qs)
        df_q["theoretical quantiles"].extend(norm_qs)
        df_dens["average cloud probability"].extend([exp_vars[i] for _ in bins])
        df_dens["bins"].extend(bins)
        df_dens["density"].extend(dens.tolist()+[0])
        if i == np.argmin(exp_vars): 
            min_res_qs = res_qs
            min_res_dens = dens.tolist()+[0]
        elif i == np.argmax(exp_vars): 
            max_res_qs = res_qs
            max_res_dens = dens.tolist()+[0]
    # plot stuff, TODO: bounds lines, sort values
    # return df_q, df_dens
    cmap = sns.color_palette("bwr")
    b, r = cmap[0], cmap[-1]
    df_dens = pd.DataFrame(df_dens).sort_values(by=["bins"])
    df_q = pd.DataFrame(df_q).sort_values(by=["theoretical quantiles"])
    fig,axs = plt.subplots(ncols=1, nrows=2, figsize=figsize)
    sns.lineplot(df_dens, x="bins", y="density", ax=axs[0], palette="bwr", hue="average cloud probability", alpha=.2)
    axs[0].set_xlim(bins[0], bins[-1])
    axs[0].legend().remove()
    axs[0].set_xlabel("value")
    axs[0].set_title("residuals distribution")
    sns.lineplot(df_q, x="theoretical quantiles", y="empirical quantiles", ax=axs[1], palette="bwr", hue="average cloud probability", alpha=.2, errorbar=None)
    axs[1].legend().remove()
    axs[1].set_title("Q-Q plot")
    # min/max lines
    for i, (x, mn, mx) in enumerate(zip([bins, norm_qs],[min_res_dens, min_res_qs], [max_res_dens, max_res_qs])):
        ax = axs[i]
        ax.plot(x, mn, color=b, label=f"minimum: {exp_vars[np.argmin(exp_vars)]:.2f}%", alpha=1)
        ax.plot(x, mx, color=r, label=f"maximum: {exp_vars[np.argmax(exp_vars)]:.2f}%", alpha=1)
        handles, labels = ax.get_legend_handles_labels()
        ax.legend([handle for i,handle in enumerate(handles[-2:])],
                  [label for i,label in enumerate(labels[-2:])],
                  title="average cloud probability",
                  loc= 'best')
    axs[0].plot(bins, dist.pdf(bins),color="k",linestyle="dotted")
    axs[1].plot(norm_qs,norm_qs,color="k",linestyle="dotted")
    # fig.suptitle(variable_name)
    # standard colorbar
    sm = plt.cm.ScalarMappable(cmap="bwr", norm=plt.Normalize(min(exp_vars), max(exp_vars)))
    axs[0].figure.colorbar(sm, ax=axs[0]) # on both here
    axs[1].figure.colorbar(sm, ax=axs[1]) 
    plt.tight_layout()
    if save_name is not None: savefigure(fig, save_name)
    plt.show()

def singlePlotNormalityAnalysis(dirs, gt_dir, variable_name, variable_index, exp_vars, figsize=(4,12), save_name=None):
    q = np.linspace(5e-4, 1-5e-4, 400)
    bins = np.linspace(-5,5,400)
    # load gt
    gtp = getPaths(dirs[0], gt_dir=gt_dir, returns=["gt"])
    gt = loadRaster(gtp, bands=variable_index, set_nan_mask=True).reshape(-1)
    if variable_index in [3,5]: gt /=100
    gt_mask = ~np.isnan(gt)
    # standardized gt
    gt_standard = (gt-np.nanmean(gt))/np.nanstd(gt)
    # gt histogram
    gt_dens = np.histogram(gt_standard, bins, density=True)[0].tolist()+[0]
    # normal distribution
    dist = scipy.stats.norm(loc=0, scale=1)
    norm_qs = [dist.ppf(qq) for qq in q]
    # load data
    df_dens = {"average cloud probability": [], "density": [], "bins": []}
    max_res_dens = None
    min_res_dens = None
    df_q = {"average cloud probability": [], "empirical quantiles": [], "theoretical quantiles": []}
    min_res_qs = None
    max_res_qs = None
    for i, d in enumerate(dirs):
        mean = loadRaster(getPaths(d, returns=["mean"]), bands=variable_index).reshape(-1)
        mask = np.bitwise_and(gt_mask, ~np.isnan(mean))
        # standardized residuals
        res = mean[mask]-gt[mask]
        res = (res-res.mean())/res.std()
        res_qs = [np.quantile(res, qq) for qq in q]
        # density
        dens, _ = np.histogram(res, bins, density=True)
        # store
        df_q["average cloud probability"].extend([exp_vars[i] for _ in q])
        df_q["empirical quantiles"].extend(res_qs)
        df_q["theoretical quantiles"].extend(norm_qs)
        df_dens["average cloud probability"].extend([exp_vars[i] for _ in bins])
        df_dens["bins"].extend(bins)
        df_dens["density"].extend(dens.tolist()+[0])
        if i == np.argmin(exp_vars): 
            min_res_qs = res_qs
            min_res_dens = dens.tolist()+[0]
        elif i == np.argmax(exp_vars): 
            max_res_qs = res_qs
            max_res_dens = dens.tolist()+[0]
    # plot stuff, TODO: bounds lines, sort values
    # return df_q, df_dens
    cmap = sns.color_palette("bwr")
    b, r = cmap[0], cmap[-1]
    df_dens = pd.DataFrame(df_dens).sort_values(by=["bins"])
    df_q = pd.DataFrame(df_q).sort_values(by=["theoretical quantiles"])
    # colorscale
    sm = plt.cm.ScalarMappable(cmap="bwr", norm=plt.Normalize(min(exp_vars), max(exp_vars)))

    # GT histogram plot
    gt_fig,gt_ax = plt.subplots(ncols=1, nrows=1, figsize=figsize)
    sns.lineplot(x=bins, y=gt_dens, ax=gt_ax)
    gt_ax.plot(bins, dist.pdf(bins),color="k",linestyle="dotted")
    gt_ax.set_xlabel("")
    gt_ax.set_ylabel("density")
    gt_ax.set_title("GT histogram")
    plt.tight_layout()
    if save_name is not None: savefigure(gt_fig, save_name+"-gt_histogram")
    plt.show()

    # Residual histogram plot
    rh_fig, rh_ax = plt.subplots(ncols=1, nrows=1, figsize=figsize)
    sns.lineplot(df_dens, x="bins", y="density", ax=rh_ax, palette="bwr", hue="average cloud probability", alpha=.2)
    rh_ax.plot(bins, dist.pdf(bins),color="k",linestyle="dotted")
    rh_ax.set_xlim(bins[0], bins[-1])
    rh_ax.legend().remove()
    rh_ax.set_xlabel("")
    rh_ax.set_title("residuals distribution")
    rh_ax.plot(bins, min_res_dens, color=b, label=f"minimum: {exp_vars[np.argmin(exp_vars)]:.2f}%", alpha=1)
    rh_ax.plot(bins, max_res_dens, color=r, label=f"maximum: {exp_vars[np.argmax(exp_vars)]:.2f}%", alpha=1)
    handles, labels = rh_ax.get_legend_handles_labels()
    rh_ax.legend([handle for i,handle in enumerate(handles[-2:])],
            [label for i,label in enumerate(labels[-2:])],
            title="average cloud probability",
            loc= 'best')
    rh_ax.figure.colorbar(sm, ax=rh_ax)
    plt.tight_layout()
    if save_name is not None: savefigure(rh_fig, save_name+"-residuals_histogram")
    plt.show()


    # Residual QQ plot
    qq_fig, qq_ax = plt.subplots(ncols=1, nrows=1, figsize=figsize)
    sns.lineplot(df_q, x="theoretical quantiles", y="empirical quantiles", ax=qq_ax, palette="bwr", hue="average cloud probability", 
                alpha=.2, errorbar=None)
    qq_ax.plot(norm_qs,norm_qs,color="k",linestyle="dotted")
    qq_ax.legend().remove()
    qq_ax.set_title("Q-Q plot")
    qq_ax.plot(norm_qs, min_res_qs, color=b, label=f"minimum: {exp_vars[np.argmin(exp_vars)]:.2f}%", alpha=1)
    qq_ax.plot(norm_qs, max_res_qs, color=r, label=f"maximum: {exp_vars[np.argmax(exp_vars)]:.2f}%", alpha=1)
    handles, labels = qq_ax.get_legend_handles_labels()
    qq_ax.legend([handle for i,handle in enumerate(handles[-2:])],
            [label for i,label in enumerate(labels[-2:])],
            title="average cloud probability",
            loc= 'best')
    qq_ax.figure.colorbar(sm, ax=qq_ax)
    plt.tight_layout()
    if save_name is not None: savefigure(qq_fig, save_name+"-qq_residuals")
    plt.show()

def getUsabilityData(
    dirs,
    s2repr_dirs,
    gt_dir,
    variable_indexes,
    variables,
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
    cps, variances, rerrors, cerrors = tuple([[] for _ in variables] for _ in range(4))
    for dir_ in dirs:
        # paths
        mean_path, variance_path, gt_path, img_path = getPaths(dir_, s2repr_dirs, gt_dir, returns=["mean","variance","gt", "img"])
        # load rasters
        gt = loadRaster(gt_path, islice=islice, jslice=jslice, set_nan_mask=True)
        gt[2] /= 100
        gt[4] /= 100
        mean = loadRaster(mean_path, bands=variable_indexes, islice=islice, jslice=jslice)
        variance = loadRaster(variance_path, bands=variable_indexes, islice=islice, jslice=jslice)
        cp = loadRaster(img_path, bands=-1, islice=islice, jslice=jslice)
        # compute valid mask
        for i in range(mean.shape[0]):
            # compute valid mask
            valid = np.bitwise_and(~np.isnan(cp), cp<100.0) # truncate 100
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
            cerrori = np.abs(rerrori-np.sqrt(variancei))
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

def conditionalHistogram(
    cps,
    Xs,
    metric,
    variable_names,
    n_bins,
    bin_cp,
    bin_fn=np.linspace,
    metric_transformation=None, 
    figsize=(12,17),
    save_name=None,
    plot_kind="histplot",
    common_norm=True
):
    fig, axs = plt.subplots(ncols=len(variable_names), nrows=1, figsize=figsize)
    for i, variable_name in enumerate(variable_names):
        cp = cps[i]
        X = Xs[i]
        if metric_transformation is not None: X = metric_transformation(X)
        if bin_cp:
            bins = bin_fn(0, 100, n_bins)
            bin_strings = [f"{bins[i]:.2f}-{bins[i+1]:.2f}" for i in range(len(bins)-1)]
            # compute bin assignments
            bin_ids = np.digitize(cp, bins=bins)
            bin_ids[bin_ids==len(bins)] = len(bins)-1
            binned = np.array(
                [bin_strings[i-1] for i in bin_ids]
            )        
            # dataframe
            df = pd.DataFrame({
                metric: X,
                "cloud probability": binned,
            })
            kwargs = dict(x=metric, hue="cloud probability")
        else:
            bins = bin_fn(X.min(), X.max(), n_bins)
            bin_strings = [f"{bins[i]:.2f}-{bins[i+1]:.2f}" for i in range(len(bins)-1)]
            # compute bin assignments
            bin_ids = np.digitize(X, bins=bins)
            bin_ids[bin_ids==len(bins)] = len(bins)-1
            binned = np.array(
                [bin_strings[i-1] for i in bin_ids]
            )        
            # dataframe
            df = pd.DataFrame({
                metric: binned,
                "cloud probability": cp,
            })
            kwargs = dict(hue=metric, x="cloud probability")
        # plot
        if plot_kind=="histplot":
            fn = sns.histplot
            kwargs.update(dict(bins=10,stat="density"))
        else: 
            fn = sns.kdeplot
        fn(
            data=df, 
            multiple="fill",
            hue_order=bin_strings,
            palette="rocket", 
            common_norm=common_norm,
            ax=axs[i],
            **kwargs
        )
        axs[i].set_xlabel("")
        axs[i].set_ylabel("Density")
        axs[i].set_title(variable_name)
        if i!=len(variable_names)-1: axs[i].get_legend().remove()
        else: sns.move_legend(axs[i], "upper left", bbox_to_anchor=(1, 1))
    for ax in axs.flat: ax.grid(False)
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
    save_name=None,
    plot_kind="boxplot",
    add_title=True,
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
        axs = axs.flatten() if num_variables>1 else [axs]
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
            df.loc[:,cloudy_visualizer.exp_var_name] = [
                bin_strings[i-1] 
                for i in bin_ids
            ]
            df = df.reset_index()
            if plot_kind=="boxplot":
                plot_fn = sns.boxplot
                plot_kwargs = dict(showfliers=False, order=bin_strings)
            elif plot_kind=="lineplot":
                plot_fn = sns.lineplot
                plot_kwargs = dict(errorbar=None)
            else: raise ValueError()
            plot_fn(
                data=df, y="x", 
                x=cloudy_visualizer.exp_var_name, 
                ax=axs[i], hue="group", 
                hue_order=["cloudless", "cloudy"],
                **plot_kwargs
            )
            ## 
            if add_title: axs[i].set_title(var)
            axs[i].set_ylabel(metric)
            if ncols>2:
                for tick in axs[i].get_xticklabels():
                    tick.set_rotation(45)
        #fig.suptitle(metric)
        if num_variables%ncols!=0: fig.delaxes(axs.flatten()[-1])
        if save_name is not None: 
            fig.tight_layout()
            savefigure(fig, save_name+f"_{metric}")
        plt.show()

