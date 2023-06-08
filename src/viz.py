import wandb
import pandas as pd
import matplotlib.pyplot as plt 
import matplotlib.patches as patches
import tempfile
import math
import json
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
            sns.boxplot(data=gvmdf, y="x", ax=ax)
        else:
            bin_strings = [f"{exp_var_bins[i]}-{exp_var_bins[i+1]}" for i in range(len(exp_var_bins)-1)]
            gvmdf.loc[:,self.exp_var_name] = [bin_strings[i-1] for i in np.digitize(gvmdf[self.exp_var_name].values, bins=exp_var_bins)]
            sns.boxplot(data=gvmdf, y="x", x=self.exp_var_name, ax=ax, order=bin_strings)
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
    
def clip(arr, bounds):
    bounds = (float(bounds[0]), float(bounds[1]))
    arr = np.where(arr>bounds[1], bounds[1], arr)
    arr = np.where(arr<bounds[0], bounds[0], arr)
    arr -= bounds[0]
    arr /= (bounds[1]-bounds[0])
    return arr

def norm2d(x, mn=None, mx=None, a=0, b=1): 
    if mn is None: mn = np.nanmin(x)
    if mx is None: mx = np.nanmax(x)
    return (b-a)*(x-mn)/(mx-mn)+a

def savefigure(fig, name):
    Path(name).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(name+".png", dpi=300)
    fig.savefig(name+".pdf", dpi=200)

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
            figsize=(12, 4), split_mask=None, save_name=None, color="g"):
    n = len(dirs)
    fig, axs = plt.subplots(ncols=len(dirs), nrows=1, figsize=figsize)
    if len(titles)>1: axs = axs.flatten()
    else: axs = [axs]
    i = 0
    for d, ax in zip(dirs, axs):
        dir_name = d.split("/")[-1]
        pid = dir_name.split("_")[0]
        img_path = list(Path(os.path.join(s2repr_dirs, dir_name, pid)).glob("*.tif"))[0]
        title = i if titles is None else titles[i]
        with rasterio.open(img_path) as f:
            rgb = f.read([4,3,2])
            rgb = clip(rgb, (100, 2000))
            rgb = rgb.transpose(1,2,0)
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
        pid = gee_dir.name.split("_")[0]
        # means
        with rasterio.open(os.path.join(gee_dir, f"{pid}_mean.tif")) as f:
            gee_mean = f.read(variable_index)
            if islice is not None: gee_mean = gee_mean[islice]
            if jslice is not None: gee_mean = gee_mean[:,jslice]
        with rasterio.open(os.path.join(orig_dir, f"{pid}_mean.tif")) as f:
            orig_mean = f.read(variable_index)
            if islice is not None: orig_mean = orig_mean[islice]
            if jslice is not None: orig_mean = orig_mean[:,jslice]
        # variances
        with rasterio.open(os.path.join(gee_dir, f"{pid}_variance.tif")) as f:
            gee_variance = f.read(variable_index)
            if islice is not None: gee_variance = gee_variance[islice]
            if jslice is not None: gee_variance = gee_variance[:,jslice]
        with rasterio.open(os.path.join(orig_dir, f"{pid}_variance.tif")) as f:
            orig_variance = f.read(variable_index)
            if islice is not None: orig_variance = orig_variance[islice]
            if jslice is not None: orig_variance = orig_variance[:,jslice]
        if normalize:
            # get stats
            meanmax, meanmin = np.nanmax(orig_mean), np.nanmin(orig_mean)
            variancemax, variancemin = np.nanmax(orig_variance), np.nanmin(orig_variance)
            # apply
            gee_mean = norm2d(gee_mean, meanmin, meanmax)
            orig_mean = norm2d(orig_mean, meanmin, meanmax)
            gee_variance = norm2d(gee_variance, variancemin, variancemax)
            orig_variance = norm2d(orig_variance, variancemin, variancemax)
        meandiff = gee_mean-orig_mean
        variancediff = gee_variance-orig_variance
        sns.heatmap(orig_mean, ax=axs[i,0], 
            vmin=min(np.nanmin(gee_mean), np.nanmean(orig_mean)), 
            vmax=max(np.nanmax(gee_mean), np.nanmax(orig_mean))
        )
        if i==0: axs[i,0].set_title(f"original mean")
        sns.heatmap(gee_mean, ax=axs[i,1], 
            vmin=min(np.nanmin(gee_mean), np.nanmean(orig_mean)), 
            vmax=max(np.nanmax(gee_mean), np.nanmax(orig_mean))
        )
        if i==0: axs[i,1].set_title(f"gee mean")
        sns.heatmap(meandiff, ax=axs[i,2], cmap="bwr", vmin=-1, vmax=1)
        if i==0: axs[i,2].set_title(f"difference mean")
        sns.heatmap(orig_variance, ax=axs[i,3], 
            vmin=min(np.nanmin(gee_variance), np.nanmean(orig_variance)), 
            vmax=max(np.nanmax(gee_variance), np.nanmax(orig_variance))
        )
        if i == 0: axs[i,3].set_title(f"original variance")
        sns.heatmap(gee_variance, ax=axs[i,4], 
            vmin=min(np.nanmin(gee_variance), np.nanmean(orig_variance)), 
            vmax=max(np.nanmax(gee_variance), np.nanmax(orig_variance))
        )
        if i == 0: axs[i,4].set_title(f"gee variance")
        sns.heatmap(variancediff, ax=axs[i,5], cmap="bwr", vmin=-1, vmax=1)
        if i == 0: axs[i,5].set_title(f"difference variance")
    for ax in axs.flatten(): 
        ax.set_xticks([])
        ax.set_yticks([])
    for i, (d, _) in enumerate(matching_dirs):
        axs[i,0].set_ylabel(datetime.strptime(d.name.split("_")[1].split("T")[0], '%Y%m%d').strftime("%d.%m.%Y"))
    fig.suptitle(variable_name)
    plt.tight_layout()
    if save_name is not None: savefigure(fig, save_name)
    plt.show()
    
def showPairedHistograms(matching_dirs, variable_index, variable_name, islice=None, jslice=None, 
                           log_mean=False, log_variance=False, save_name=None, figsize=(10, 15)):
    fig, axs = plt.subplots(nrows=len(matching_dirs), ncols=2, figsize=figsize)
    for i, (orig_dir, gee_dir) in enumerate(matching_dirs):
        pid = gee_dir.name.split("_")[0]
        # means
        with rasterio.open(os.path.join(gee_dir, f"{pid}_mean.tif")) as f:
            gee_mean = f.read(variable_index)
            if islice is not None and jslice is not None: gee_mean = gee_mean[islice, jslice]
            gee_mean = gee_mean.flatten()
        with rasterio.open(os.path.join(orig_dir, f"{pid}_mean.tif")) as f:
            orig_mean = f.read(variable_index)
            if islice is not None and jslice is not None: orig_mean = orig_mean[islice, jslice]
            orig_mean = orig_mean.flatten()
        # variances
        with rasterio.open(os.path.join(gee_dir, f"{pid}_variance.tif")) as f:
            gee_variance = f.read(variable_index)
            if islice is not None and jslice is not None: gee_variance = gee_variance[islice, jslice]
            gee_variance = gee_variance.flatten()
        with rasterio.open(os.path.join(orig_dir, f"{pid}_variance.tif")) as f:
            orig_variance = f.read(variable_index)
            if islice is not None and jslice is not None: orig_variance = orig_variance[islice, jslice]
            orig_variance = orig_variance.flatten()
        km = "mean" if not log_mean else "log mean"
        mdf = pd.DataFrame({
            "source": ["original" for _ in orig_mean]+["gee" for _ in gee_mean],
            km: orig_mean.tolist()+gee_mean.tolist()
        })
        kv = "variance" if not log_variance else "log variance"
        vdf = pd.DataFrame({
            "source": ["original" for _ in orig_variance]+["gee" for _ in gee_variance],
            kv: orig_variance.tolist()+gee_variance.tolist()
        })
        sns.histplot(data=mdf, x=km, hue="source", ax=axs[i,0])
        axs[i,0].set_title(datetime.strptime(orig_dir.name.split("_")[1].split("T")[0], '%Y%m%d').strftime("%d.%m.%Y"))
        if log_mean: axs[i,0].set_xscale("log")
        sns.histplot(data=vdf, x=kv, hue="source", ax=axs[i,1])
        axs[i,1].set_title(datetime.strptime(orig_dir.name.split("_")[1].split("T")[0], '%Y%m%d').strftime("%d.%m.%Y"))
        axs[i,1].set_ylabel("")
        if log_variance: axs[i,1].set_xscale("log")
        axs[i,0].get_legend().remove()
        axs[i,1].get_legend().remove()
    fig.suptitle(variable_name)
    fig.legend([axs[0,0], axs[0,1]], labels=["original", "gee"], ncol=2, loc='upper center', bbox_to_anchor=(0.5, 0.))
    plt.tight_layout()
    if save_name is not None: savefigure(fig, save_name)
    plt.show()

def showMeanDifferenceMaps(orig_path, gee_path, 
                           variable_index, variable_name,
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
    with rasterio.open(orig_path) as f:
        orig_mean = f.read(variable_index)
    with rasterio.open(gee_path) as f:
        gee_mean = f.read(variable_index)
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
            # get stats
            mx, mn = np.nanmax(O), np.nanmin(O)
            # apply
            O = norm2d(O, mn, mx)
            G = norm2d(G, mn, mx)
        rerror = G-O
        sns.heatmap(
            rerror,
            cmap="bwr",
            vmin=-1, 
            vmax=1,
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
    
def showPredictionMaps(dirs, titles, variable_index, variable_name, s2repr_dirs, gt_dir, 
                        shapefile_paths, islice=None, jslice=None, normalize=False, save_name=None,
                        figsize=(15,10)):
    fig, axs = plt.subplots(nrows=len(titles), ncols=6, figsize=figsize)
    for i, d in enumerate(dirs):
        dir_name = d.split("/")[-1]
        pid = dir_name.split("_")[0]
        img_path = list(Path(os.path.join(s2repr_dirs, dir_name, pid)).glob("*.tif"))[0]
        mean_path = Path(os.path.join(d, f"{pid}_mean.tif"))
        variance_path = Path(os.path.join(d, f"{pid}_variance.tif"))
        gt_path = os.path.join(gt_dir, f"{pid}.tif")
        gt_date = compute_gt_date(pid, shapefile_paths)
        with rasterio.open(img_path) as f:
            rgb = f.read([4,3,2])
            rgb = clip(rgb, (100, 2000))
            rgb = rgb.transpose(1,2,0)
            if islice is not None: rgb = rgb[islice]
            if jslice is not None: rgb = rgb[:,jslice]
        with rasterio.open(gt_path) as f:
            gt = f.read(variable_index)
            gt_mask = f.read_masks(1)//255
            gt[gt_mask==0]=np.nan
            if variable_index in [3,5]: gt /= 100 # Cover/Dens normalization!!
            if islice is not None: gt = gt[islice]
            if jslice is not None: gt = gt[:,jslice]
        with rasterio.open(mean_path) as f:
            mean = f.read(variable_index)
            if islice is not None: mean = mean[islice]
            if jslice is not None: mean = mean[:,jslice]
        with rasterio.open(variance_path) as f:
            variance = f.read(variable_index)
            if islice is not None: variance = variance[islice]
            if jslice is not None: variance = variance[:,jslice]
        if normalize:
            # get stats
            rgbmax, rgbmin = np.nanmax(rgb, axis=(0,1)), np.nanmin(rgb, axis=(0,1))
            gtmax, gtmin = np.nanmax(gt), np.nanmin(gt)
            variancemax, variancemin = np.nanmax(variance), np.nanmin(variance)
            # apply
            rgb = norm2d(rgb, rgbmin, rgbmax)
            gt = norm2d(gt, gtmin, gtmax)
            mean = norm2d(mean, gtmin, gtmax)
            variance = norm2d(variance, variancemin, variancemax)
        rerror = mean-gt
        cerror = np.abs(rerror)-np.sqrt(variance)
        axs[i,0].imshow(rgb)
        if i ==0: axs[i,0].set_title(f"RGB")
        sns.heatmap(gt, ax=axs[i,1], 
            vmin=min(np.nanmin(mean), np.nanmean(gt)), 
            vmax=max(np.nanmax(mean), np.nanmax(gt))
        )
        if i==0: axs[i,1].set_title(f"gt ({gt_date.strftime('%d.%m.%Y')})")
        # break
        sns.heatmap(mean, ax=axs[i,2], 
            vmin=min(np.nanmin(mean), np.nanmean(gt)), 
            vmax=max(np.nanmax(mean), np.nanmax(gt))
        )
        if i==0: axs[i,2].set_title(f"mean")
        sns.heatmap(variance, ax=axs[i,3], vmin=np.nanmin(variance), vmax=np.nanmax(variance))
        if i==0: axs[i,3].set_title(f"variance")
        sns.heatmap(rerror, ax=axs[i,4], cmap="bwr", vmin=-1, vmax=1)
        if i==0: axs[i,4].set_title(f"r-error")
        sns.heatmap(cerror, ax=axs[i,5], cmap="bwr", vmin=-1, vmax=1)
        if i==0: axs[i,5].set_title(f"c-error")
    for ax in axs.flatten(): 
        ax.set_xticks([])
        ax.set_yticks([])
    for i, d, in enumerate(dirs):
        axs[i,0].set_ylabel(f'{titles[i]} ({datetime.strptime(Path(d).name.split("_")[1].split("T")[0], "%Y%m%d").strftime("%d.%m.%Y")})')
    fig.suptitle(variable_name)
    plt.tight_layout()
    if save_name is not None: savefigure(fig, save_name)
    plt.show()

def showPredictionHistograms(dirs, titles, variable_index, variable_name,
                           gt_dir, shapefile_paths, log_mean=False, log_variance=False,
                           islice=None, jslice=None,
                           save_name=None, figsize=(10, 15)):
    fig, axs = plt.subplots(nrows=len(titles), ncols=2, figsize=figsize)
    for i, d in enumerate(dirs):
        dir_name = d.split("/")[-1]
        pid = dir_name.split("_")[0]
        mean_path = Path(os.path.join(d, f"{pid}_mean.tif"))
        variance_path = Path(os.path.join(d, f"{pid}_variance.tif"))
        gt_path = os.path.join(gt_dir, f"{pid}.tif")
        gt_date = compute_gt_date(pid, shapefile_paths)
        with rasterio.open(gt_path) as f:
            gt = f.read(variable_index)
            gt_mask = f.read_masks(1)//255
            gt[gt_mask==0]=np.nan
            if variable_index in [3,5]: gt /= 100 # Cover/Dens normalization!!
            if islice is not None: gt = gt[islice]
            if jslice is not None: gt = gt[:,jslice]
        with rasterio.open(mean_path) as f:
            mean = f.read(variable_index)
            if islice is not None: mean = mean[islice]
            if jslice is not None: mean = mean[:,jslice]
        with rasterio.open(variance_path) as f:
            variance = f.read(variable_index)
            if islice is not None: variance = variance[islice]
            if jslice is not None: variance = variance[:,jslice]
        km = "mean" if not log_mean else "log mean"
        mdf = pd.DataFrame({
            "source": [f"gt ({gt_date.strftime('%d.%m.%Y')})" for _ in gt.flatten()]+
                      [f'prediction ({datetime.strptime(Path(d).name.split("_")[1].split("T")[0], "%Y%m%d").strftime("%d.%m.%Y")})' for _ in mean.flatten()],
            km: gt.flatten().tolist()+mean.flatten().tolist()
        })
        kv = "variance" if not log_variance else "log variance"
        vdf = pd.DataFrame({
            "source": ["prediction" for _ in variance.flatten()],
            kv: variance.flatten().tolist()
        })
        axs[i,0] = sns.histplot(data=mdf, x=km, hue="source", ax=axs[i,0])
        sns.move_legend(axs[i,0], "upper center", ncol=2, title="", fontsize="small", bbox_to_anchor=(0.5,1.15))
        if log_mean: axs[i,0].set_xscale("log")
        sns.histplot(data=vdf, x=kv, ax=axs[i,1])
        # axs[i,1].set_title(titles[i])
        axs[i,1].set_ylabel("")
        if log_variance: axs[i,1].set_xscale("log")
    for i, d, in enumerate(dirs):
        axs[i,0].set_ylabel(f'{titles[i]}\nCount')
    fig.suptitle(variable_name)
    plt.tight_layout()
    if save_name is not None: savefigure(fig, save_name)
    plt.show()

def showErrorHistograms(dirs, titles, variable_index, variable_name,
                           gt_dir, log_rerror=False, log_cerror=False, 
                           islice=None, jslice=None,
                           save_name=None, figsize=(10,15)):
    fig, axs = plt.subplots(nrows=len(titles), ncols=2, figsize=figsize)
    for i, d in enumerate(dirs):
        dir_name = d.split("/")[-1]
        pid = dir_name.split("_")[0]
        mean_path = Path(os.path.join(d, f"{pid}_mean.tif"))
        variance_path = Path(os.path.join(d, f"{pid}_variance.tif"))
        gt_path = os.path.join(gt_dir, f"{pid}.tif")
        with rasterio.open(gt_path) as f:
            gt = f.read(variable_index)
            gt_mask = f.read_masks(1)//255
            gt[gt_mask==0]=np.nan
            if variable_index in [3,5]: gt /= 100 # Cover/Dens normalization!!
            if islice is not None: gt = gt[islice]
            if jslice is not None: gt = gt[:,jslice]
        with rasterio.open(mean_path) as f:
            mean = f.read(variable_index)
            if islice is not None: mean = mean[islice]
            if jslice is not None: mean = mean[:,jslice]
        with rasterio.open(variance_path) as f:
            variance = f.read(variable_index)
            if islice is not None: variance = variance[islice]
            if jslice is not None: variance = variance[:,jslice]
        rerror = mean-gt
        cerror = np.sqrt(rerror**2)-np.sqrt(variance)
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
        sns.histplot(data=rdf, x=rk, ax=axs[i,0])
        # axs[i,0].set_title(titles[i])
        if log_rerror: axs[i,0].set_xscale("log")
        sns.histplot(data=cdf, x=ck, ax=axs[i,1])
        # axs[i,1].set_title(titles[i])
        axs[i,1].set_ylabel("")
        if log_cerror: axs[i,1].set_xscale("log")
    for i, d, in enumerate(dirs):
        axs[i,0].set_ylabel(f'{titles[i]}\nCount')
    fig.suptitle(variable_name)
    plt.tight_layout()
    if save_name is not None: savefigure(fig, save_name)
    plt.show()
    
def showSinglePredictionMaps(dir_, title, variable_index, variable_name, s2repr_dirs, gt_dir, 
                                islice=None, jslice=None, normalize=False, save_name=None,
                                figsize=(15,2.5)):
    fig, axs = plt.subplots(nrows=1, ncols=6, figsize=figsize)
    dir_name = dir_.split("/")[-1]
    pid = dir_name.split("_")[0]
    img_path = list(Path(os.path.join(s2repr_dirs, dir_name, pid)).glob("*.tif"))[0]
    mean_path = Path(os.path.join(dir_, f"{pid}_mean.tif"))
    variance_path = Path(os.path.join(dir_, f"{pid}_variance.tif"))
    gt_path = os.path.join(gt_dir, f"{pid}.tif")
    with rasterio.open(img_path) as f:
        rgb = f.read([4,3,2])
        rgb = clip(rgb, (100, 2000))
        rgb = rgb.transpose(1,2,0)
        if islice is not None: rgb = rgb[islice]
        if jslice is not None: rgb = rgb[:,jslice]
    with rasterio.open(gt_path) as f:
        gt = f.read(variable_index)
        gt_mask = f.read_masks(1)//255
        gt[gt_mask==0]=np.nan
        if variable_index in [3,5]: gt /= 100 # Cover/Dens normalization!!
        if islice is not None: gt = gt[islice]
        if jslice is not None: gt = gt[:,jslice]
    with rasterio.open(mean_path) as f:
        mean = f.read(variable_index)
        if islice is not None: mean = mean[islice]
        if jslice is not None: mean = mean[:,jslice]
    with rasterio.open(variance_path) as f:
        variance = f.read(variable_index)
        if islice is not None: variance = variance[islice]
        if jslice is not None: variance = variance[:,jslice]
    if normalize:
        # get stats
        rgbmax, rgbmin = np.nanmax(rgb, axis=(0,1)), np.nanmin(rgb, axis=(0,1))
        gtmax, gtmin = np.nanmax(gt), np.nanmin(gt)
        variancemax, variancemin = np.nanmax(variance), np.nanmin(variance)
        # apply
        rgb = norm2d(rgb, rgbmin, rgbmax)
        gt = norm2d(gt, gtmin, gtmax)
        mean = norm2d(mean, gtmin, gtmax)
        variance = norm2d(variance, variancemin, variancemax)
    rerror = mean-gt
    cerror = np.abs(rerror)-np.sqrt(variance)
    axs[0].imshow(rgb)
    axs[0].set_title(f"rgb")
    sns.heatmap(gt, ax=axs[1], 
        vmin=min(np.nanmin(mean), np.nanmean(gt)), 
        vmax=max(np.nanmax(mean), np.nanmax(gt))
    )
    axs[1].set_title(f"gt")
    # break
    sns.heatmap(mean, ax=axs[2], 
        vmin=min(np.nanmin(mean), np.nanmean(gt)), 
        vmax=max(np.nanmax(mean), np.nanmax(gt))
    )
    axs[2].set_title(f"mean")
    sns.heatmap(variance, ax=axs[3], vmin=np.nanmin(variance), vmax=np.nanmax(variance))
    axs[3].set_title(f"variance")
    sns.heatmap(rerror, ax=axs[4], cmap="bwr", vmin=-1, vmax=1)
    axs[4].set_title(f"r-error")
    sns.heatmap(cerror, ax=axs[5], cmap="bwr", vmin=-1, vmax=1)
    axs[5].set_title(f"c-error")
    for ax in axs.flatten(): ax.set_axis_off()
    fig.suptitle(f"{variable_name} - {title}")
    plt.tight_layout()
    if save_name is not None: savefigure(fig, save_name)
    plt.show()
    
def CompareMapsStats(matching_dirs, variable_names, split_mask=None, split=None):
    mean_stats = {"variable": [], "stat": [], "original": [], "gee": [], "delta": [], "relative delta (%)": []}
    variance_stats = {"variable": [], "stat": [], "original": [], "gee": [], "delta": [], "relative delta (%)": []}
    if split_mask is not None or split is not None: 
        assert split_mask is not None and split is not None
        split_ = split
        split = ["train", "val", "test"].index(split)
        mean_stats["split"] = []
        variance_stats["split"] = []    
    for i, (orig_dir, gee_dir) in enumerate(matching_dirs):
        pid = gee_dir.name.split("_")[0]
        # means
        with rasterio.open(os.path.join(gee_dir, f"{pid}_mean.tif")) as f:
            n = f.count
            gee_mean = f.read(f.indexes)
            if split_mask is not None: gee_mean = gee_mean[:,split_mask==split]
            else: gee_mean = gee_mean.reshape(gee_mean.shape[0], -1)
        with rasterio.open(os.path.join(orig_dir, f"{pid}_mean.tif")) as f:
            orig_mean = f.read(f.indexes)
            if split_mask is not None: orig_mean = orig_mean[:,split_mask==split]
            else: orig_mean = orig_mean.reshape(orig_mean.shape[0], -1)
        with rasterio.open(os.path.join(gee_dir, f"{pid}_variance.tif")) as f:
            gee_variance = f.read(f.indexes)
            if split_mask is not None: gee_variance = gee_variance[:,split_mask==split]
            else: gee_variance = gee_variance.reshape(gee_variance.shape[0], -1)
        with rasterio.open(os.path.join(orig_dir, f"{pid}_variance.tif")) as f:
            orig_variance = f.read(f.indexes)
            if split_mask is not None: orig_variance = orig_variance[:,split_mask==split]
            else: orig_variance = orig_variance.reshape(orig_variance.shape[0], -1)
        for f, fn in zip([np.nanmin, np.nanmax, np.nanmean, np.nanstd, np.nanmedian], 
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
            variance_stats["variable"].extend(variable_names)
            variance_stats["stat"].extend([fn for _ in range(1, n+1)])
            variance_stats["original"].extend(f(orig_variance, axis=1).tolist())
            variance_stats["gee"].extend(f(gee_variance, axis=1).tolist())
            variance_stats["delta"].extend((f(gee_variance, axis=1)-f(orig_variance, axis=1)).tolist())
            variance_stats["relative delta (%)"].extend(
                ((f(gee_variance, axis=1)-f(orig_variance, axis=1))/f(orig_variance, axis=1)*100).tolist()
            )
            if split is not None:
                variance_stats["split"].extend([split_ for _ in range(1, n+1)])
        return pd.DataFrame(mean_stats), pd.DataFrame(variance_stats)
    
def evaluateSplit(prediction_dir, gt_dir, split_mask, split):
    split_id = ["train", "val", "test"].index(split)
    # Load data
    vpath = list(Path(prediction_dir).glob("*_variance.tif"))[0]
    project = vpath.stem.split("_")[0]
    with rasterio.open(vpath) as f:
        variance = f.read(f.indexes)
        variance[:,split_mask!=split_id] = np.nan
    with rasterio.open(os.path.join(prediction_dir, f"{project}_mean.tif")) as f:
        mean = f.read(f.indexes)
        mean[:,split_mask!=split_id] = np.nan
    with rasterio.open(os.path.join(gt_dir, f"{project}.tif")) as f:
        gt = f.read(f.indexes)
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
    rcu.add_project(project, gt, mean, variance)
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
            data["imageId"].extend([gee_dir.name, orig_dir.name])
            data["source"].extend(["gee", "original"])
            data["x"].extend([gt.x, ot.x])
            if split: data["split"].extend([split, split])
    fulldf = pd.DataFrame(data)
    return fulldf

def plotTrainTestMetricsDataFrames(train_df, test_df, type="bar", save_name=None, figsize=(15,10)):
    assert type in ["bar", "scatter"]
    metrics_df = pd.concat([train_df, test_df])
    fig, axs = plt.subplots(
        nrows=len(metrics_df.metric.unique()),
        ncols=len(metrics_df.variable.unique()),
        figsize=figsize
    )
    for i, m in enumerate(metrics_df.metric.unique()):
        for j, v in enumerate(metrics_df.variable.unique()):
            tmp = (metrics_df
                   .query(f"metric == '{m}' & variable == '{v}'")
                   .drop(columns=["metric", "variable", "imageId"]))
            mtmp = (tmp
               .groupby(by=["source", "split"])
               .mean()
               .reset_index())
            if type == "scatter":
                sns.scatterplot(data=tmp, x="split", y="x", hue="source", ax=axs[i,j], marker="o", s=20, alpha=0.4)
                sns.scatterplot(data=mtmp, x="split", y="x", hue="source", ax=axs[i,j], marker="_", s=100, linewidth=2)
                axs[i,j].set(xlim=(-0.5, 1.5))
            else:
                sns.barplot(data=tmp, x="split", y="x", hue="source", ax=axs[i,j], 
                            hue_order=["original", "gee"], errorbar=None)
            axs[i,j].get_legend().remove()
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