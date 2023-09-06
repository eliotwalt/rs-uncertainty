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
if [[ "$MACHINE" == "--euler" ]]  || [[ "$MACHINE" == "--local" ]]; then
  echo "Running on: $MACHINE"
else
  echo "invalid machine"; exit 1
fi

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
  echo "Setting up env"
  PYVERSION=3.7.4
  GDALVESION=3.4.3
  module load python/$PYVERSION gdal/$GDALVERSION
  pip install gdal==`gdal-config --version` --user # do not put it in requirements, the version must match local install!
  pip install -r requirements.txt --user
  retvalue=($(python /cluster/work/igp_psr/elwalt/pdm/rs-uncertainty/src/scripts/create_dataset.py --configure ${ARGS[@]:1}))
else 
  retvalue=($(python /scratch/ewalt/pdm/rs-uncertainty/src/scripts/create_dataset.py --configure ${ARGS[@]:1}))
fi
CFG=${retvalue[0]}
sub_cfgs=(${retvalue[@]:1})
echo "Saved main config file : $CFG"
echo "Saved sub config files : $(dirname $CFG)"

# *(2)* Submit preprocessing jobs
if [[ $MACHINE == "--euler" ]];
then
  echo "Preparing preprocessing job array ..."
  # compute slurm options as they are not static and can't be defined with #SBATCH in script
  echo "Computing job array options ...."
  num_jobs=${#sub_cfgs[@]}
  timestamp=$(basename $(dirname $(dirname ${sub_cfgs[0]})))
  time=$(python -c "from datetime import timedelta; td=timedelta(hours=2)*$NUM_PROJECTS_PER_JOB; hours, minutes = td.seconds//3600, (td.seconds//60)%60; print(f'{hours:0>2}:{minutes:0>2}:00')")
  log_dir=/cluster/work/igp_psr/elwalt/logs/dataset/$timestamp
  job_name="preprocess_projects"
  log_file=$log_dir/pp_%a-$num_jobs.log
  echo "Array job name : $job_name"
  echo "number of jobs : $num_jobs"
  echo "Log file       : $log_file"
  echo "Compute time   : $time"
  options="-n 1 --time $time --mem-per-cpu=64000 --job-name=$job_name --array=1-$num_jobs --output=$log_file --error=$log_file"
  echo "Options        : $options"
  # create log dir/file
  mkdir -p $log_dir
  touch $log_file 
  # submit job
  echo "Submitting job array ..."
  retvalue=($(sbatch $options /cluster/work/igp_psr/elwalt/pdm/rs-uncertainty/run/slurm/preprocess_projects.sh ${sub_cfgs[@]}))
  echo "${retvalue[@]}"
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
  echo "Preparing aggregation job ..."
  # compute slurm options as they are not static and can't be defined with #SBATCH in script
  echo "Computing job options ...."
  timestamp=$(python -c "import yaml; f=open('$CFG', 'r'); data=yaml.safe_load(f); f.close(); print(data['save_dir'].split('/')[-1])")
  log_dir=/cluster/work/igp_psr/elwalt/logs/dataset/$timestamp
  job_name=aggregate_projects
  log_file=$log_dir/agg.log
  echo "Job name       : $job_name"
  echo "Log file       : $log_file"
  options="-n 1 --mem-per-cpu=32000 --time=4:00:00 --depend=afterok:$pp_job_array_id --job-name=$job_name --output=$log_file --error=$log_file"
  echo "Options        : $options"
  # create log dir/file
  mkdir -p $(dirname $log_file)
  touch $log_file
  retvalue=($(sbatch $options /cluster/work/igp_psr/elwalt/pdm/rs-uncertainty/run/slurm/aggregate_projects.sh $CFG))
  echo "${retvalue[@]}"
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
if [[ $MACHINE == "--euler" ]];
then
  job_ids=(`scontrol show jobid -dd $pp_job_array_id | grep --color "JobId=" | cut -d " " -f 1 | cut -d "=" -f 2`)
  echo "Job ids:"
  echo "- aggregation job array   : $agg_job_id"
  echo "- preprocessing job array : $pp_job_array_id"
  echo "Get array job job ids: scontrol show jobid -dd $pp_job_array_id | grep --color 'JobId=' | cut -d ' ' -f 1 | cut -d '=' -f 2"
  echo "Monitor status with: sacct -n -X -j $pp_job_array_id"
fi

# # *(6)* kill jobs
# if [[ $MACHINE == "--euler" ]];
# then
#   echo "[debug] Cancelling jobs ..."
#   scancel $pp_job_array_id $agg_job_id
#   echo "Done."
# fi
