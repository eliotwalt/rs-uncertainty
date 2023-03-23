#!/bin/bash

# ************************************************** HELPSTRING **************************************************
read -r -d '' HELP <<HELPSTRING
Create dataset on Euler
This script generates and submits batch jobs on Euler to generate a dataset. Creation is done as a job array while
statistic aggregation is performed once all scripts have finished.
Usage:
    -c | --cfg                  : path to configuration file
    -n | --num_projects_per_job : number of projects in each array job
HELPSTRING

# ************************************************** ARGUMENTS ***************************************************
# source: https://stackabuse.com/how-to-parse-command-line-arguments-in-bash/
ARGS=("$@")
if [[ ${#ARGS[@]} == 0 ]]; then
  echo "Ivalid options. Use -h or --help for help."
  exit 1
fi
SHORT=,c:,s:,h
LONG=cfg:,num_projects_per_job:,help
OPTS=$(getopt -a -n dataset_creation --options $SHORT --longoption $LONG -- "$@")
eval set -- "$OPTS"
while :
do
  case "$1" in
    --cfg )
      CFG="$2"
      shift 2
      ;;
    --num_projects_per_job )
      NUM_PROJECTS_PER_JOB="$2"
      shift 2
      ;;
    -h | --help )
      printf "%s\n" "$HELP"
      shift;
      exit 0
      ;;
    --)
      shift;
      break
      ;;
    *)
      echo "Unexpected option: $1"
      echo "Use -h or --help for more information"
      exit 1
      ;;
  esac
done
echo "Main config file     : $CFG"
echo "Num projects per job : $NUM_PROJECTS_PER_JOB"

# **************************************************** LOGIC ****************************************************
# *(1)* Configure 
read -r -a sub_cfgs <<< `python /cluster/work/igp_psr/elwalt/pdm/rs-uncertainty/src/scripts/create_dataset.py --configure ${ARGS[@]}`
CFG=${sub_cfgs[0]}
sub_cfgs=${sub_cfgs[@]:1}
echo "Saved sub config files: $(dirname $CFG)"

# *(2)* Submit preprocessing jobs
echo "Submitting preprocessing job array ..."
retvalue=($(sbatch /cluster/work/igp_psr/elwalt/pdm/rs-uncertainty/run/slurm/preprocess_projects.sh ${sub_cfgs[@]}))
pp_job_array_id=${retvalue[-1]}
echo "Done."

# *(3)* Submit aggregation job
echo "Submitting aggregation job ..."
retvalue=($(sbatch /cluster/work/igp_psr/elwalt/pdm/rs-uncertainty/run/slurm/aggregate_projects.sh $CFG $pp_job_array_id))
agg_job_id=${retvalue[-1]}
echo "Done."

# *(4)* Delete sub config files
echo "Deleting sub config files ..."
glob=$(dirname $CFG)/*-*-*-*-*/
rm -r $glob
echo "Done."

# *(5)* Feedback
echo "Job ids:"
echo "- preprocessing job array : $pp_job_array_id"
echo "- aggregation job array   : $agg_job_id"

# *(6)* kill jobs
echo "[debug] Cancelling jobs ..."
scancel $pp_job_array_id $agg_job_id
echo "Done."
