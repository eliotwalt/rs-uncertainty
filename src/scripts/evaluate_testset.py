import os, sys
root = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, root)
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
    """
    x load standardization data
    x loop on projects, compute variance bounds online
    x init rcu
    - loop on projects
        - standardize
        - add project
        - get([project_id])
    - loop on regions
        - get(region)
    - save results (incl. histogram)
    """
    results = {}
    # Load config
    args = parse_args()
    print(f"Loading config file {args.cfg}...")
    with args.cfg.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
        for k, v in cfg.items():
            if k.endswith("_dir"): cfg[k] = Path(v)
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
    rcu.get_results_df(
        groups={"east": cfg["projects_east"], "west": cfg["projects_west"], "north": cfg["projects_north"]},
        variable_names=cfg["variable_names"]
    )
    rcu.save_json(pjoin(cfg["prediction_dir"], "rcu.json"))

if __name__ == "__main__": main()
