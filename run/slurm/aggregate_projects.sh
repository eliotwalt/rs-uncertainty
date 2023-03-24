#!/bin/bash

echo "Starting aggregation job"

# ************************************************** ARGUMENTS ***************************************************
CONFIG_FILE=$1
JOB_ARRAY_ID=$2

# ************************************************* ENVIRONMENT **************************************************
echo "Setting up env"
PYVERSION=3.7.4
GDALVESION=3.4.3
module load python/$PYVERSION gdal/$GDALVERSION
pip install gdal==`gdal-config --version` --user # do not put it in requirements, the version must match local install!
pip install -r requirements.txt --user

# **************************************************** GLOBAL ****************************************************
TIMESTAMP=$(python -c "import yaml; f=open('$CONFIG_FILE', 'r'); data=yaml.safe_load(f); f.close(); print(data['save_dir'].split('/')[-1])")
LOG_DIR=/cluster/work/igp_psr/elwalt/logs/aggregate_projects
JOB_NAME=aggregate_projects
LOG_FILE=$LOG_DIR/$TIMESTAMP/std.log
echo "Job name       : $JOB_NAME"
echo "Log file       : $LOG_FILE"
# create log dir/file
mkdir -p $(dirname $LOG_FILE)
touch $LOG_FILE

# **************************************************** SLURM *****************************************************
#SBATCH -n 1
#SBATCH --time=15:00
#SBATCH --mem-per-cpu=3000
#SBATCH --depend=afterok:$JOB_ARRAY_ID
#SBATCH --job-name=$JOB_NAME
#SBATCH --output=$LOG_FILE
#SBATCH --error=$LOG_FILE

python /cluster/work/igp_psr/elwalt/pdm/rs-uncertainty/src/scripts/create_dataset.py --aggregate --cfg $CONFIG_FILE