import yaml, os, argparse
from pathlib import Path
from gee_download import GEELocalDownloader
from utils import get_gt_project_data, get_s2_project_data

p = argparse.ArgumentParser()
p.add_argument("-c", "--cfg", help="config file", required=True)
args = p.parse_args()
with open(args.cfg) as f: 
    cfg = yaml.safe_load(f)

if cfg["project_data_source"] == "s2": get_project_data = get_s2_project_data
elif cfg["project_data_source"] == "gt": get_project_data = get_gt_project_data
else: raise ValueError(f"'project_data_source' must be in ['gt', 's2']")

# downloader
gee_cfg = cfg["gee"]
gee_cfg["verbose"] = cfg["verbose"]
geedl = GEELocalDownloader(**gee_cfg)

# download
ts_cfg = cfg["timeserie"]
localdir = os.path.join(ts_cfg["localdir"], "original")
reprojected_localdir = os.path.join(ts_cfg["localdir"], "reprojected")
for i, project in enumerate(cfg["projects"]):
    # project data
    data_cfg = cfg["data"]
    data_cfg["project_id"] = project
    if cfg["project_data_source"] == "s2": 
        data_cfg["s2_date"] = data_cfg["s2_date"][i]
    if cfg["verbose"]: print("Loading data for project", project)
    ref_path, _, ref_date, _, bbox = get_project_data(**data_cfg)
    projectdir = os.path.join(localdir, project)
    Path(projectdir).mkdir(parents=True, exist_ok=True)
    reprojected_projectdir = os.path.join(reprojected_localdir, project)
    Path(reprojected_projectdir).mkdir(parents=True, exist_ok=True)
    ts_cfg.update({
        "project_id": project,
        "localdir": projectdir,
        "reprojected_localdir": reprojected_projectdir,
        "bbox": bbox,
        "ref_date": ref_date,
        "ref_path": ref_path,
    })
    geedl.download_timeserie(**ts_cfg)


