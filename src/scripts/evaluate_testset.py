import os, sys
root = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, root)
import rasterio
import argparse
import yaml
import numpy as np
from tqdm import tqdm
from pathlib import Path
from rcu_metrics import StratifiedRCU

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
    labels_mean = np.array(stats["labels_stats"]["mean"])
    labels_std = np.array(stats["labels_stats"]["std"])
    # loop on projects to get variance bounds
    print(f"Computing variance bounds in {cfg['pkl_dir']}...")
    lo_variance = np.full((5,), np.inf)
    hi_variance = np.full((5,), -np.inf)
    # for variance_file in tqdm(list(cfg["prediction_dir"].glob("*_variance.tif"))):
    variance_files = []
    for variance_file in tqdm(list(cfg["prediction_dir"].glob("*_variance.tif"))[:3]): # Debug
        if variance_file.stem.split("_")[0] not in projects: continue
        variance_files.append(variance_file)
        with rasterio.open(variance_file) as fh:
            variance = fh.read(fh.indexes)
        variance_flat = variance.reshape(5, -1)
        hi = np.nanmax(variance_flat, axis=1)
        lo = np.nanmin(variance_flat, axis=1)
        hi_variance[hi>hi_variance] = hi[hi>hi_variance]
        lo_variance[lo<lo_variance] = lo[lo<lo_variance]
    # initialize RCU metrics
    print("Initiating StratifiedRCU object...")
    rcu = StratifiedRCU(
        num_variables=len(cfg["data_bands"]),
        # num_groups=len(projects),
        num_groups=3, # Debug
        num_bins=cfg["num_bins"],
        lo_variance=lo_variance,
        hi_variance=hi_variance
    )
    # compute stats online
    print(f"Computing stats online from in {cfg['prediction_dir']}...")
    for variance_file in tqdm(variance_files):
        # load data
        project = mean_file.stem.split('_')[0]
        if project not in projects: continue
        with rasterio.open(pjoin(cfg['prediction_dir'], f"{project}_mean.tif")) as fh:
            mean = fh.read(fh.indexes)
        with rasterio.open(variance_file) as fh:
            variance = fh.read(fh.indexes)
        with rasterio.open(pjoin(cfg['gt_dir'], f"{project}.tif")) as fh:
            gt = fh.read(fh.indexes)
        # standardize
        variance /= (labels_std)**2
        mean = (mean-labels_mean)/labels_std
        gt = (gt-labels_mean)/labels_std 
        # add project
        rcu.add_project(project, gt, mean, variance)
        # get project metrics
        results[project] = rcu.get(project)
    # compute regions
    # print(f"Aggregegating metrics...")
    # for region in ["east", "west", "north"]:
    #     region_projects = cfg[f"projects_{region}"]
    #     results[region] = rcu.get_subset(region_projects) # Debug
    # compute all
    print(f"Aggregating across all projects...")
    results["global"] = rcu.get_all()
    # write results
    # res_file = pjoin(cfg["prediction_dir"], "metrics.yaml")
    res_file = Path("ignore.debug-eval.metrics.yaml")
    print(f"writing results to {res_file}...")
    with res_file.open("w", encoding="utf-8") as f:
        yaml.dump(results, f, sort_keys=False)
    print("Done.")

if __name__ == "__main__": main()
