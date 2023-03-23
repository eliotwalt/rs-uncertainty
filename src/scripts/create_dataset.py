import os, sys
from pathlib import Path
import argparse
import yaml
from datetime import datetime
from copy import deepcopy

def path(x):
    try:
        return Path(x)
    except Exception as e: raise e

def pjoin(*subs): return Path(os.path.abspath(os.path.join(*subs)))

def configure(cfg_f, num_project_per_job):
    """
    Given a blowtorch configuration and a number of jobs, configure each jobs (i.e generate job configurations
    and return them). Each job gets a list of projects to process.
    """
    # load main configuration
    with cfg_f.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    # create save_dir
    save_dir = pjoin(cfg["save_dir"], datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    save_dir.mkdir(parents=True)
    cfg["save_dir"] = str(save_dir)
    # confugre jobs
    projects = cfg["projects"]
    sub_cfg_paths = []
    for i in range(0, len(projects), num_project_per_job):
        sub_projects = projects[i:i+num_project_per_job]
        cfg["projects"] = sub_projects
        sub_cfg_path = pjoin(save_dir, cfg_f.stem+"_sub_"+str(i)+".yaml")
        with sub_cfg_path.open("w", encoding="utf-8") as f:
            yaml.dump(cfg, f, sort_keys=False)
        sub_cfg_paths.append(sub_cfg_path)
    print(" ".join([str(p) for p in sub_cfg_paths]))
    
def aggregate():
    """
    Given a path to a directory of processed projects, aggreagate the statistics
    """
    pass

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    # configuration args
    p.add_argument("--configure", help="set to configure mode", action="store_true")
    p.add_argument("--cfg", help="path to config file (yaml)", type=path)
    p.add_argument("--num_project_per_job", help="number of jobs", type=int)
    # aggregation args
    p.add_argument("--aggregate", help="set to aggregate mode", action="store_true")
    args = p.parse_args()
    if args.configure:
        assert args.num_project_per_job, "--num_project_per_job must be set in configuration mode"
        configure(args.cfg, args.num_project_per_job)
    elif args.aggregate:
        aggregate(args.save_dir)
    else: raise AttributeError("Must be set either to configuration (--configure) or aggregatation (--aggregate) mode.")
    sys.exit(0)