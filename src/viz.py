import wandb
import pandas as pd
import matplotlib.pyplot as plt 
import tempfile
import json
import os
import seaborn as sns
import numpy as np
from .metrics import StratifiedRCU
sns.set()
sns.set_style("whitegrid")

class ExperimentVisualizer():
    def __init__(self, rcus, exp_var_name, exp_vars, variable_names):
        self.exp_var_name = exp_var_name 
        self.exp_vars = exp_vars
        self.rcus = rcus
        self.variable_names = variable_names
        self.df = self._make_df()

    def _make_df(self):
        for i, df in enumerate(self.rcus):
            self.rcus[i].results[self.exp_var_name] = [self.exp_vars[i] for _ in range(self.rcus[i].results.shape[0])]
        return pd.concat([r.results for r in self.rcus])

    @classmethod
    def from_paths(cls, paths, exp_var_name, exp_vars, variable_names):
        assert isinstance(paths, list)
        rcus = [StratifiedRCU.from_json(os.path.join(p, "rcu.json")) for p in paths]
        return cls(rcus, exp_var_name, exp_vars, variable_names)
    
    @classmethod
    def from_wandb(cls, *args, **kwargs):
        raise NotImplementedError()
    
    def variable_histogram_plot(self, variable, ax=None, hi_bound=np.inf, log=True):
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
        g = sns.lineplot(data=hhdf, x="variance", hue=self.exp_var_name, y="probability", ax=ax)
        if log: _ = g.set(xscale='log', xlabel="log variance")
        return ax
    
    def histogram_plot(self, hi_bound=np.inf, log=True):
        if not isinstance(hi_bound, list): hi_bound = [hi_bound for _ in range(len(self.variable_names))]
        num_variables = len(self.variable_names)
        ncols = num_variables//2
        nrows = num_variables-ncols
        fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(12, 10))
        axs = axs.flatten()
        for i, var in enumerate(self.variable_names):
            self.variable_histogram_plot(var, axs[i], hi_bound[i], log)
            axs[i].set_title(var)
            axs[i].legend(loc="upper right")
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
        return ax

    def metric_plot(self, metric, kind):
        num_variables = len(self.variable_names)
        ncols = num_variables//2
        nrows = num_variables-ncols
        fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(12, 10))
        axs = axs.flatten()
        for i, var in enumerate(self.variable_names):
            self.variable_metric_plot(metric, var, kind, axs[i])
            axs[i].set_title(var)
        fig.suptitle(metric)
        return axs
    
    def variable_calibration_plot(self, metric, variable, k=100, ax=None):
        if ax is None:
            fig, ax = plt.subplots()
        var_idx = self.variable_names.index(variable)
        assert metric in ["uce", "ence", "auce"]
        if metric == "uce": cols = ["bin variance", "bin mse", self.exp_var_name]
        elif metric == "auce": cols = ["expected accuracy", "empirical accuracy", self.exp_var_name]
        else: cols = ["bin std", "bin rmse", self.exp_var_name]
        ccdf = pd.DataFrame(columns=cols)
        for i, rcu in enumerate(self.rcus):
            lorcu = rcu.upsample(k)
            xc, yc = lorcu.get_calibration_curve(metric)
            if metric != "auce": xc = xc[var_idx]
            ct = np.full(xc.shape, str(self.exp_vars[i]))
            ccdf = pd.concat([ccdf, pd.DataFrame({cols[0]:xc, cols[1]:yc[var_idx], cols[-1]:ct[var_idx]})])
        ccdf = ccdf.dropna()
        lo = min(ccdf[cols[0]].min(), ccdf[cols[1]].min())
        hi = max(ccdf[cols[0]].max(), ccdf[cols[1]].max())
        id_line = (
            np.linspace(lo, hi, 2), 
            np.linspace(lo, hi, 2)
        )
        ax.plot(*id_line, color="black", linestyle="dotted")
        sns.lineplot(data=ccdf, x=cols[0], y=cols[1], hue=cols[2], ax=ax)
        return ax
    
    def calibration_plot(self, metric, k=100):
        num_variables = len(self.variable_names)
        ncols = num_variables//2
        nrows = num_variables-ncols
        fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(12, 12))
        axs = axs.flatten()
        for i, var in enumerate(self.variable_names):
            self.variable_calibration_plot(metric, var, k=k, ax=axs[i])
            axs[i].set_title(var)
        fig.suptitle(metric+" calibration plot")
        return axs


#     def plot_metric(self, metric: str, ax):
#         pass

#     def plot_ause(self, metric: str, ax):
#         pass

#     def plot_(): pass

# class LocalExperimentVisualizer():
#     def __init__(
#         self,
#         paths,
#         add_baseline: bool=True
#     ):
#         self.add_baseline = add_baseline
#         if self.add_baseline: self.add_baseline: self.baseline_rcu, self.rcus = self._get_rcus()
#         else: else: self.rcus = self._get_rcus()
#         self.df = self._build_df()

#     def 

# class WandBExperimentVisualizer():
#     def __init__(
#         self,
#         entity: str,
#         project: str,
#         filter_tag: str,
#         variable_name: str,
#         add_baseline: bool=True,
#     ):
#         self.root = f"{entity}/{project}"
#         self.entiy = entity
#         self.project = project
#         self.variable_name = variable_name
#         self.filter_tag = filter_tag
#         self.add_baseline = add_baseline
#         if self.add_baseline: self.baseline, self.runs = self._get_runs()
#         else: self.runs = self._get_runs()
#         if self.add_baseline: self.baseline_rcu, self.rcus = self._get_rcus()
#         else: self.rcus = self._get_rcus()
#         self.df = self._build_df()
    
#     def _get_runs(self):
#         api = wandb.Api()
#         if self.add_baseline:
#             baselines = api.runs(path=self.root, filters={"tags": "baseline"})
#             assert len(baselines)==1, "Multiple baseline runs were found: {}".format(", ".join(b.name for b in baselines))
#         variable_runs = api.runs(path=self.root, filters={"tags": self.filter_tag})
#         print(f"Found {len(variable_runs)} runs for experiment on variable {self.variable_name}.")
#         if self.add_baseline: return baselines[0], variable_runs
#         else: return variable_runs
    
#     def _get_rcus(self):
#         def download_rcu(run, path):
#             with tempfile.TemporaryDirectory() as d:
#                 with run.file(path).download(d) as f:
#                     data = json.load(f)
#             return StratifiedRCU.from_json(data)
#         if self.add_baseline:
#             path = list(filter(lambda x: x.name.endswith("rcu.json"), list(self.baseline.file)))[0]
#             baseline_rcu = download_rcu(self.baseline, path)
#         rcus = []
#         for run in self.runs:
#             path = list(filter(lambda x: x.name.endswith("rcu.json"), list(run.file)))[0]
#             # download json
#             rcus.append(download_rcu(run, path))
#         if self.add_baseline: return baseline_rcu, rcus
#         else: return rcus
    
#     # def _build_df(self):
#     #     def fill_run(run, df_dict, experiment):
#     #         for key, value in run.summary.items():
#     #             try:
#     #                 kind, metric, variable, group = key.split("-")
#     #                 df_dict["experiment"].append(experiment)
#     #                 df_dict[self.variable_name].append(run.config[self.variable_name])
#     #                 df_dict["kind"].append(kind)
#     #                 df_dict["metric"].append(metric)
#     #                 df_dict["variable"].append(variable)
#     #                 df_dict["group"].append(group)
#     #                 df_dict["value"].append(value)
#     #             except:
#     #                 pass
#     #         return df_dict
#     #     dict_df = { "experiment": [],self.variable_name: [],"kind": [],"metric": [],"variable": [],"group": [],"value": []}
#     #     for run in self.runs: dict_df = fill_run(run, dict_df, "cloud_threshold")
#     #     if self.add_baseline: dict_df = fill_run(self.baseline, dict_df, f"baseline ({self.baseline.config[self.variable_name]})")      
#     #     return pd.DataFrame(dict_df)

    
# class CloudThresholdVisualizer(ExperimentVisualizer):
#     def __init__(
#         self,
#         entity: str,
#         project: str,
#         add_baseline: bool=True,
#     ): 
#         super().__init__(
#             entity, project, 
#             filter_tag="cloud_threshold", 
#             variable_name="dataset.cloudy_pixels_threshold", 
#             add_baseline=add_baseline
#         )

