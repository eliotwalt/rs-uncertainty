#!/bin/bash

echo "Starting prediction job"

# ************************************************** ARGUMENTS ***************************************************
CREATE_CONFIG=$1
PREDICT_CONFIG=$2
EVAL_CONFIG=$3

# ************************************************* ENVIRONMENT **************************************************
echo "Setting up env"
PYVERSION=3.7.4
GDALVESION=3.4.3
module load python/$PYVERSION gdal/$GDALVERSION
pip install gdal==`gdal-config --version` --user # do not put it in requirements, the version must match local install!
pip install -r requirements.txt --user

# **************************************************** COMMAND ***************************************************
set -e 
echo "***** creating dataset *****"
python /cluster/work/igp_psr/elwalt/pdm/rs-uncertainty/src/scripts/create_dataset.py --preprocess --cfg CREATE_CONFIG
echo "***** predicting testset *****"
python /cluster/work/igp_psr/elwalt/pdm/rs-uncertainty/src/scripts/predict_testset.py --cfg PREDICT_CONFIG
echo "***** evaluating testset *****"
python /cluster/work/igp_psr/elwalt/pdm/rs-uncertainty/src/scripts/evaluate_testset.py --cfg EVAL_CONFIG
set +e