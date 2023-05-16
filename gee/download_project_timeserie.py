import yaml, os, argparse
from pathlib import Path
from gee_download import GEELocalDownloader
from utils import get_project_data

p = argparse.ArgumentParser()
p.add_argument("-c", "--cfg", help="config file", required=True)
args = p.parse_args()
with open(args.cfg) as f: 
    cfg = yaml.safe_load(f)

# downloader
gee_cfg = cfg["gee"]
gee_cfg["verbose"] = cfg["verbose"]
geedl = GEELocalDownloader(**gee_cfg)

# download
ts_cfg = cfg["timeserie"]
localdir = ts_cfg["localdir"]
for project in cfg["projects"]:
    # project data
    data_cfg = cfg["data"]
    data_cfg["project_id"] = project
    if cfg["verbose"]: print("Loading data for project ", project)
    gt_path, _, _, gt_date, gt_crs, _, bbox = get_project_data(**data_cfg)
    projectdir = os.path.join(localdir, project)
    Path(projectdir).mkdir(parents=True, exist_ok=True)
    ts_cfg.update({
        "project_id": project,
        "localdir": projectdir,
        "bbox": bbox,
        "gt_date": gt_date,
        "gt_path": gt_path,
    })
    geedl.download_timeserie(**ts_cfg)


