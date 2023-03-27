#!/bin/bash

echo "Starting preprocessing job array"

# ************************************************** ARGUMENTS ***************************************************
CONFIG_FILES=("$@")

# ************************************************* ENVIRONMENT **************************************************
echo "Setting up env"
PYVERSION=3.7.4
GDALVESION=3.4.3
module load python/$PYVERSION gdal/$GDALVERSION
pip install gdal==`gdal-config --version` --user # do not put it in requirements, the version must match local install!
pip install -r requirements.txt --user

# **************************************************** COMMAND ***************************************************
i=$((${SLURM_ARRAY_TASK_ID}-1))
python /cluster/work/igp_psr/elwalt/pdm/rs-uncertainty/src/scripts/create_dataset.py --preprocess --cfg ${CONFIG_FILES[$i]}
