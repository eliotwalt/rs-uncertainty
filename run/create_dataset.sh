#!/bin/bash

# ************************************************** HELPSTRING **************************************************
read -r -d '' HELP <<HELPSTRING
Create dataset (on Euler or locally)
This script generates and submits jobs on Euler or run them locally to generate a dataset. Creation is done as a job
array on Euler and in parallel locally while statistic aggregation is performed once all scripts have finished.
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
MACHINE=$1
echo $MACHINE
if [[ $MACHINE != "--euler"  ||  $MACHINE != "--local" ]]; then
    echo "invalid machine"; exit 1
fi
echo "Running on: $MACHINE"
SHORT=,c:,s:,h
LONG=cfg:,num_projects_per_job:,help
OPTS=$(getopt -a -n dataset_creation --options $SHORT --longoption $LONG -- "${@:2}")
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
if [[ $MACHINE == "--euler" ]]; then
  retvalue=($(python /cluster/work/igp_psr/elwalt/pdm/rs-uncertainty/src/scripts/create_dataset.py --configure ${ARGS[@]:1}))
else 
  retvalue=($(python /scratch/ewalt/pdm/rs-uncertainty/src/scripts/create_dataset.py --configure ${ARGS[@]:1}))
fi
CFG=${retvalue[0]}
sub_cfgs=(${retvalue[@]:1})
echo "Saved sub config files: $(dirname $CFG)"

# *(2)* Submit preprocessing jobs
if [[ $MACHINE == "--euler" ]];
then
  echo "Submitting preprocessing job array ..."
  retvalue=($(sbatch /cluster/work/igp_psr/elwalt/pdm/rs-uncertainty/run/slurm/preprocess_projects.sh ${sub_cfgs[@]}))
  pp_job_array_id=${retvalue[-1]}
  echo "Done."
else
  pp_pids=()
  echo "Launching preprocessing processes ..."
  for (( i=0; i<${#sub_cfgs[@]}; i++ ));
  do
    echo "Launching ${sub_cfgs[$i]}"
    python /scratch/ewalt/pdm/rs-uncertainty/src/scripts/create_dataset.py --preprocess --cfg ${sub_cfgs[$i]} 1> dataset.$i.log 2> dataset.$i.log &
    pids[$i]=$!
  done
  echo "Done. Check logs: dataset.[0-$(($i-1))].log"
  echo "waiting on ${pids[@]} ..."
  for pid in "${pids[@]}";
  do
    wait $pid
  done
  echo "Done."
fi

# *(3)* Submit aggregation job
if [[ $MACHINE == "--euler" ]];
then
  echo "Submitting aggregation job ..."
  retvalue=($(sbatch /cluster/work/igp_psr/elwalt/pdm/rs-uncertainty/run/slurm/aggregate_projects.sh $CFG $pp_job_array_id))
  agg_job_id=${retvalue[-1]}
  echo "Done."
else
  echo "Launching aggregation process ..."
  python /scratch/ewalt/pdm/rs-uncertainty/src/scripts/create_dataset.py --aggregate --cfg $CFG
  pid=$!
  wait $pid
  echo "Done."
  echo "Waiting on $pid ..."
  echo "Done."
fi

# *(5)* Feedback
if [[ $MACHINE == "euler" ]];
then
  echo "Job ids:"
  echo "- preprocessing job array : $pp_job_array_id"
  echo "- aggregation job array   : $agg_job_id"
fi

# *(6)* kill jobs
if [[ $MACHINE == "euler" ]];
then
  echo "[debug] Cancelling jobs ..."
  scancel $pp_job_array_id $agg_job_id
  echo "Done."
fi
