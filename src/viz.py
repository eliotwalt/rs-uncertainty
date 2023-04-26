import wandb
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns
sns.set()
sns.set_style("whitegrid")

class ExperimentVisualizer():
    def __init__(
        self,
        entity: str,
        project: str,
        filter_tag: str,
        variable_name: str,
        add_baseline: bool=True,
    ):
        self.root = f"{entity}/{project}"
        self.entiy = entity
        self.project = project
        self.variable_name = variable_name
        self.filter_tag = filter_tag
        self.add_baseline = add_baseline
        if self.add_baseline: self.baseline, self.runs = self._get_runs()
        else: self.runs = self._get_runs()
        self.df = self._build_df()
    
    def _get_runs(self):
        ret = []
        api = wandb.Api()
        if self.add_baseline:
            baselines = api.runs(path=self.root, filters={"tags": "baseline"})
            assert len(baselines)==1, "Multiple baseline runs were found: {}".format(", ".join(b.name for b in baselines))
            ret.append(baselines[0])
        variable_runs = api.runs(path=self.root, filters={"tags": self.filter_tag})
        print(f"Found {len(variable_runs)} runs for experiment on variable {self.variable_name}.")
        ret.append(variable_runs)
        return tuple(ret)
    
    def _build_df(self):
        def fill_run(run, df_dict, experiment):
            for key, value in run.summary.items():
                try:
                    kind, metric, variable, group = key.split("-")
                    df_dict["experiment"].append(experiment)
                    df_dict[self.variable_name].append(run.config[self.variable_name])
                    df_dict["kind"].append(kind)
                    df_dict["metric"].append(metric)
                    df_dict["variable"].append(variable)
                    df_dict["group"].append(group)
                    df_dict["value"].append(value)
                except:
                    pass
            return df_dict
        dict_df = { "experiment": [],self.variable_name: [],"kind": [],"metric": [],"variable": [],"group": [],"value": []}
        for run in self.runs: dict_df = fill_run(run, dict_df, "cloud_threshold")
        if self.add_baseline: dict_df = fill_run(self.baseline, dict_df, f"baseline ({self.baseline.config[self.variable_name]})")      
        return pd.DataFrame(dict_df)

    def plot_metric(self, metric: str, ax):
        pass

    def plot_ause(self, metric: str, ax):
        pass

    def plot_

    
class CloudThresholdVisualizer(ExperimentVisualizer):
    def __init__(
        self,
        entity: str,
        project: str,
        add_baseline: bool=True,
    ): super().__init__(entity, project, filter_tag="cloud_threshold", variable_name="dataset.cloudy_pixels_threshold", add_baseline=add_baseline)