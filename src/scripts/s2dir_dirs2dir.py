import os 
from pathlib import Path
import sys

input_dir = sys.argv[1]
output_dir = sys.argv[2]
Path(output_dir).mkdir(parents=True, exist_ok=True)

# loop on project ids
input_dir = os.path.abspath(input_dir)
for project_dir in os.scandir(input_dir):
    # make sure it is a project id dir
    if not os.path.isdir(project_dir.path): continue
    if not project_dir.name.isnumeric(): continue
    # loop on images
    for img_path in Path(project_dir.path).glob("*.tif"):
        imageId = project_dir.name+"_"+img_path.stem.split('_')[3]
        destination = os.path.join(output_dir, imageId, project_dir.name, img_path.name)
        Path(destination).parents[0].mkdir(parents=True, exist_ok=True)
        os.symlink(img_path, destination)       
        print(f"Created symlink {img_path} -> {destination}")