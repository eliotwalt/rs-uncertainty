#!/bin/bash

# ************************************************** ARGUMENTS ***************************************************
CONFIG_FILES=("$@")

# **************************************************** GLOBAL ****************************************************
NUM_JOBS=${#CONFIG_FILES[@]}
TIMESTAMP=$(basename $(dirname ${CONFIG_FILES[0]}))
TIME=$(python -c "from datetime import timedelta; dt=timedelta(hours=12); print(dt/$NUM_JOBS)")
LOG_DIR=/cluster/work/igp_psr/elwalt/logs/preprocess_projects/$TIMESTAMP
ID_SUFFIX=$SLURM_ARRAY_TASK_ID-$SLURM_ARRAY_TASK_COUNT
JOB_NAME=preprocess_projects_$ID_SUFFIX
LOG_FILE=$LOG_DIR/$ID_SUFFIX.log
PYVERSION=3.7.4
GDALVESION=3.4.3

# **************************************************** SLURM *****************************************************
# #SBATCH -n 1
# #SBATCH --time=$TIME
# #SBATCH --mem-per-cpu=8000
# #SBATCH --array=1-$NUM_JOBS
# #SBATCH --job-name=$JOB_NAME
# #SBATCH --output=$LOG_FILE
# #SBATCH --error=$LOG_FILE

# # Environment
# module load python/$PYVERSION gdal/$GDALVERSION
# pip install gdal=`gdal-config --version` --user # do not put it in requirements, the version must match local install!
# pip install -r requirements.txt --user

# python /scratch/ewalt/pdm/rs-uncertainty/src/scripts/create_dataset.py --preprocess --cfg $CONFIG_FILES[$SLURM_ARRAY_TASK_ID]