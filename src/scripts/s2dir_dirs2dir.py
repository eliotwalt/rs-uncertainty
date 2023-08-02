import os 
from pathlib import Path
import sys
import shutil

"""
USAGE:  python src/scripts/s2dir_dirs2dir.py assets/data/sentinel_data/s2_reprojected/ gee_data/${experiment_name}/ ${desired project id, e.g. 1023}
"""

input_dir = sys.argv[1]
output_dir = sys.argv[2]
try:
    project_id = sys.argv[3]
except IndexError:
    project_id = None
Path(output_dir).mkdir(parents=True, exist_ok=True)

# loop on project ids
input_dir = os.path.abspath(input_dir)
for project_dir in os.scandir(input_dir):
    # make sure it is a project id dir
    if not os.path.isdir(project_dir.path): continue
    if not project_dir.name.isnumeric(): continue
    # make sure it is the requested project_id
    if project_id is not None: 
        if not project_id==project_dir.name: continue
    # loop on images
    for img_path in Path(project_dir.path).glob("*.tif"):
        imageId = project_dir.name+"_"+img_path.stem.split('_')[3]
        destination = os.path.join(output_dir, imageId, project_dir.name, img_path.name)
        Path(destination).parents[0].mkdir(parents=True, exist_ok=True)
        shutil.copyfile(img_path, destination)       
        print(f"Copied {img_path} -> {destination}")