import os, sys
root = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, root)
import wandb
from datetime import datetime
import rasterio
import argparse
import yaml
import numpy as np
from tqdm import tqdm
from pathlib import Path
from metrics import StratifiedRCU

def _path(x):
    try:
        return Path(x)
    except Exception as e: raise e

def pjoin(*subs): return Path(os.path.abspath(os.path.join(*subs)))

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--cfg", help="Path to prediction config file", required=True, type=_path)
    return p.parse_args()

def main():
    results = {}
    # Load configs
    args = parse_args()
    print(f"Loading config file {args.cfg}...")
    with args.cfg.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
        for k, v in cfg.items():
            if k.endswith("_dir"): cfg[k] = Path(v)
    with open(os.path.join(cfg["pkl_dir"], "data_config.yaml")) as f:
        dataset_cfg = yaml.safe_load(f)
    with open(os.path.join(cfg["prediction_dir"], "prediction_config.yaml")) as f:
        prediction_cfg = yaml.safe_load(f)
    config = {}
    for prefix, cfg in zip(["dataset", "prediction", "evaluation"], [dataset_cfg, prediction_cfg, cfg]):
        for key, values in cfg.items():
            config[f"{prefix}.{key}"] = values
    # intialize wandb
    wb_run = wandb.init(
        project="rcu-evaluation",
        config=config,
        name=os.path.basename(cfg["prediction_dir"]),
        tags=cfg["tags"]
    )
    # define projects span
    projects = cfg["projects_east"]+cfg["projects_west"]+cfg["projects_north"]
    # Load standardization data
    with pjoin(cfg["pkl_dir"], "stats.yaml").open("r", encoding="utf-8") as f:
        stats = yaml.safe_load(f)
    labels_mean = np.array(stats["labels_stats"]["mean"]).reshape(5,1,1)
    labels_std = np.array(stats["labels_stats"]["std"]).reshape(5,1,1)
    # loop on projects to get variance bounds
    print(f"Computing variance bounds in {cfg['pkl_dir']}...")
    lo_variance = np.full((5,), np.inf)
    hi_variance = np.full((5,), -np.inf)
    variance_files = []
    for variance_file in tqdm(list(cfg["prediction_dir"].glob("*_variance.tif"))):
        if variance_file.stem.split("_")[0] not in projects: continue
        variance_files.append(variance_file)
        with rasterio.open(variance_file) as fh:
            variance = fh.read(fh.indexes)
        variance_flat = variance.reshape(5, -1)
        hi = np.nanmax(variance_flat, axis=1)
        lo = np.nanmin(variance_flat, axis=1)
        hi_variance[hi>hi_variance] = hi[hi>hi_variance]
        lo_variance[lo<lo_variance] = lo[lo<lo_variance]
    print("Variances lower bound:", lo_variance.tolist())
    print("Variances upper bound:", hi_variance.tolist())
    # initialize RCU metrics
    print("Initiating StratifiedRCU object...")
    rcu = StratifiedRCU(
        num_variables=len(cfg["data_bands"]),
        num_groups=len(variance_files),
        num_bins=cfg["num_bins"],
        lo_variance=lo_variance,
        hi_variance=hi_variance
    )
    # compute stats online
    print(f"Computing stats online from predictions: {cfg['prediction_dir']}...")
    for variance_file in tqdm(variance_files):
        # load data
        project = variance_file.stem.split('_')[0]
        if project not in projects: continue
        with rasterio.open(pjoin(cfg['prediction_dir'], f"{project}_mean.tif")) as fh:
            mean = fh.read(fh.indexes)
        with rasterio.open(variance_file) as fh:
            variance = fh.read(fh.indexes)
        with rasterio.open(pjoin(cfg['gt_dir'], f"{project}.tif")) as fh:
            gt = fh.read(fh.indexes)
            gt[2] /= 100 # Cover/Dens normalization!!
            gt[4] /= 100
        # standardize meanH and p95
        if cfg["normalize_mean"]:
            mean[[0,1]] = (mean[[0,1]]-labels_mean[[0,1]])/labels_std[[0,1]]
            gt[[0,1]] = (gt[[0,1]]-labels_mean[[0,1]])/labels_std[[0,1]]
        if cfg["normalize_variance"]:
            variance /= labels_std**2
        # add project
        rcu.add_project(project, gt, mean, variance)
    print(f"Computing result dataframe")
    rcu.get_results_df(
        groups={"east": cfg["projects_east"], "west": cfg["projects_west"], "north": cfg["projects_north"]},
        variable_names=cfg["variable_names"]
    )
    print(f"Serializing StratifiedRCU object to "+str(pjoin(cfg["prediction_dir"], "rcu.json"))+"...")
    rcu.save_json(pjoin(cfg["prediction_dir"], "rcu.json"))
    # log rcu json
    print(f"Logging serialized StratifiedRCU...")
    wb_run.save(str(pjoin(cfg["prediction_dir"], "rcu.json")))
    # log metrics for each group, variable and kind
    log_df = rcu.results.copy()
    log_df["key"] = log_df.apply(lambda x: "-".join([x["kind"], x["metric"], x["variable"], x["group"]]), axis=1)
    log_df = log_df[["key", "x"]]
    print(f"Logging {len(log_df)} metrics...")
    wb_run.log({key: value for key, value in zip(log_df.key, log_df.x)})
    # close
    print("Finishing run...")
    wb_run.finish()

if __name__ == "__main__": main()
