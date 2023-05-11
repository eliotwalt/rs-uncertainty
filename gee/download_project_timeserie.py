import argparse
import yaml
from ee_download_workflow import download_project
from utils import get_project_data

p = argparse.ArgumentParser()
p.add_argument("-c", "--cfg", help="config file", required=True)
args = p.parse_args()
with open(args.cfg) as f: 
    cfg = yaml.safe_load(f)
data_cfg = {"project_id": cfg["project_id"]}
for key in ["gt_dir", "shapefile_paths", "gt_data_bands"]: data_cfg[key] = cfg.pop(key)
gt_file, gt, gt_date, gt_crs, polygon, _ = get_project_data(**data_cfg)
cfg["polygon"] = polygon
cfg["crs"] = gt_crs
cfg["gt_date"] = gt_date
download_project(**cfg, verbose=True)
