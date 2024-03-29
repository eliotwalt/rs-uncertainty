#!/bin/bash

echo "Starting training"

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
python /cluster/work/igp_psr/elwalt/pdm/rs-uncertainty/src/train.py $CONFIG_FILE