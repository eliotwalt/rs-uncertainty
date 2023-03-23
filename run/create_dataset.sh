#!/bin/bash

# ************************************************** HELPSTRING **************************************************
read -r -d '' HELP << HELPSTRING
Create dataset on Euler
This script generates and submits batch jobs on Euler to generate a dataset. Creation is done as a job array while
statistic aggregation is performed once all scripts have finished.
Usage:
    -j | --job_name   : Slurm job name
    -c | --cfg        : path to configuration file
    -a | --array_size : number of projects in each array job
HELPSTRING

# ************************************************** ARGUMENTS ***************************************************
# source: https://stackabuse.com/how-to-parse-command-line-arguments-in-bash/
SHORT=j:,c:,s:,h
LONG=job_name:,cfg:,array_size:,help
OPTS=$(getopt -a -n dataset_creation --options $SHORT --longoption $LONG -- "$@")
eval set -- "$OPTS"
while :
do
  case "$1" in
    -j | --job_name )
      JOB_NAME="$2"
      shift 2
      ;;
    -c | --cfg )
      CFG="$2"
      shift 2
      ;;
    -n | --num_projects_per_job )
      NUM_PROJECTS_PER_JOB="$2"
      shift 2
      ;;
    -h | --help )
      printf "%s\n" "$HELP"
      shift;
      exit 1
      ;;
    --)
      shift;
      break
      ;;
    *)
      echo "Unexpected option: $1"
      ;;
  esac
done
echo "Scheduling: $JOB_NAME"
echo "Main config file: $CFG"
echo "Job array size: $ARRAY_SIZE"

# **************************************************** LOGIC ****************************************************
# *(1)* Configure 
read -r -a sub_cfgs <<< `python /cluster/work/igp_psr/elwalt/pdm/rs-uncertainty/src/scripts/create_dataset.py 
    --configure --cfg $CFG ----num_projects_per_job $NUM_PROJECTS_PER_JOB`
tmp=$(basename ${sub_cfgs[0]})
tmp=$(dirname ${sub_cfgs[0]})/`echo $tmp | sed 's/[0-9]/*/g'`
echo "Saved sub config files: $tmp"

# *(2)* Submit preprocessing jobs
pp_job_array_id=$(sbatch ./slurm/preprocess_projects_arr.sh ${sub_cfgs[@]})

# *(3)* Submit aggregation job
agg_job_id=$(sbatch ./slurm/aggregate_projects.sh $CFG $pp_job_array_id)

# *(4)* Feedback
echo "Successfully submitted preprocessing job array ($pp_job_array_id) and aggregation job ($agg_job_id)"