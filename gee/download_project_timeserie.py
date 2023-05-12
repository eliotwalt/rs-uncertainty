import yaml, os, argparse
from pathlib import Path
from gee_download import GoogleEarthEngineLocalDownloader
from utils import get_project_data

p = argparse.ArgumentParser()
p.add_argument("-c", "--cfg", help="config file", required=True)
args = p.parse_args()
with open(args.cfg) as f: 
    cfg = yaml.safe_load(f)

# project data
data_cfg = cfg["data"]
data_cfg["project_id"] = cfg["project_id"]
if cfg["verbose"]: print("Loading data for project ", cfg["project_id"])
gt_file, _, gt_date, gt_crs, polygon, _ = get_project_data(**data_cfg)

# downloader
gee_cfg = cfg["gee"]
gee_cfg["verbose"] = cfg["verbose"]
geedl = GoogleEarthEngineLocalDownloader(**gee_cfg)

# download
ts_cfg = cfg["timeserie"]
ts_cfg["project_id"] = cfg["project_id"]
ts_cfg["localdir"] = os.path.join(ts_cfg["localdir"], ts_cfg["project_id"])
Path(ts_cfg["localdir"]).mkdir(parents=True, exist_ok=True)
ts_cfg.update({
    "polygon": polygon,
    "crs": gt_crs, 
    "gt_date": gt_date
})
geedl.download_timeserie(**ts_cfg)


