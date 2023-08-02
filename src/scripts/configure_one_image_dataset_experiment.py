# create temporary config files from template configuration files for 
# cloud experiment
import yaml, os
from datetime import datetime
from pathlib import Path
from copy import deepcopy
import shutil
import argparse

NOW = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
NOW_TOKEN = "${now}"
IMAGEID_TOKEN = "${imageId}"

def check_integrity(create_config, predict_config, eval_config):
    assert create_config["s2_reprojected_dir"]==predict_config["s2_reprojected_dir"]
    assert create_config["gt_dir"]==predict_config["gt_dir"]==eval_config["gt_dir"]
    assert create_config["name"]==predict_config["name"]
    assert create_config["save_dir"]==predict_config["pkl_dir"]==eval_config["pkl_dir"]
    assert create_config["sampling_strategy"]==predict_config["sampling_strategy"]
    assert os.path.exists(create_config["stats_path"])
    assert predict_config["save_dir"]==eval_config["prediction_dir"]
    assert predict_config["closest_s1"]==True
    assert predict_config["add_date_to_save_dir"]==False
    if "stats_path" in create_config or "stats_path" in predict_config: 
        assert create_config["stats_path"] == predict_config["stats_path"]

def check_tokens(*configs):
    for config in configs:
        for key, value in config.items():
            if key.endswith("_dir") and isinstance(value, str):
                assert NOW_TOKEN not in value.split("/"), f"found NOW_TOKEN in {key}: {value}"
                assert IMAGEID_TOKEN not in value.split("/"), f"found IMAGEID_TOKEN in {key}: {value}"

def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--create", help="create dataset config template")
    p.add_argument("--predict", help="predict testset config template")
    p.add_argument("--eval", help="eval testset config template")
    p.add_argument("--tmp_dir", help="tmp dir", default="/tmp", required=False)
    args = p.parse_args()
    return args.create, args.predict, args.eval, args.tmp_dir

if __name__ == "__main__":
    create_path, predict_path, eval_path, tmp_dir = get_args()
    with open(create_path) as f:
        create_config = yaml.safe_load(f)
    with open(predict_path) as f:
        predict_config = yaml.safe_load(f)
    with open(eval_path) as f:
        eval_config = yaml.safe_load(f)
    check_integrity(create_config, predict_config, eval_config)
    # Loop on single image dirs
    for imageDir in os.scandir(Path(create_config["s2_reprojected_dir"]).parents[0]):
        td = Path(f"{tmp_dir}/1imgds_{imageDir.name}")
        if os.path.isdir(td): shutil.rmtree(td)
        td.mkdir()
        subcreate_config, subpredict_config, subeval_config = deepcopy(create_config), deepcopy(predict_config), deepcopy(eval_config)
        configs = [subcreate_config, subpredict_config, subeval_config]
        for i, config in enumerate(configs):
            for key, value in config.items():
                if key.endswith("_dir") and isinstance(value, str):
                    value = value.replace(NOW_TOKEN, NOW)
                    value = value.replace(IMAGEID_TOKEN, imageDir.name)
                    configs[i][key] = value
        check_integrity(subcreate_config, subpredict_config, subeval_config)
        check_tokens(subcreate_config, subpredict_config, subeval_config)
        paths = []
        with open(os.path.join(td, "create_config.yaml"), "w") as f:
            paths.append(f.name)
            yaml.dump(subcreate_config, f)
        with open(os.path.join(td, "predict_config.yaml"), "w") as f:
            paths.append(f.name)
            yaml.dump(subpredict_config, f)
        with open(os.path.join(td, "eval_config.yaml"), "w") as f:
            paths.append(f.name)
            yaml.dump(subeval_config, f)
        print(" ".join(paths))
