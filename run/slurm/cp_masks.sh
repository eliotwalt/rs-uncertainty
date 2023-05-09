#!/bin/bash

echo "Starting cp masks creation job"

# ************************************************** ARGUMENTS ***************************************************
CONFIG_FILE=$1

# ************************************************* ENVIRONMENT **************************************************
echo "Setting up env"
PYVERSION=3.7.4
GDALVESION=3.4.3
module load python/$PYVERSION gdal/$GDALVERSION
pip install gdal==`gdal-config --version` --user # do not put it in requirements, the version must match local install!
pip install -r requirements.txt --user

# **************************************************** COMMAND ***************************************************
python /cluster/work/igp_psr/elwalt/pdm/rs-uncertainty/src/scripts/create_cp_masks.py --cfg $CONFIG_FILE