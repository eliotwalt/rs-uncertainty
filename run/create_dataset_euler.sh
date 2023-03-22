#!/bin/bash

# ************************************************** HELPSTRING **************************************************
read -r -d '' HELP << HELPSTRING
Create dataset on Euler.
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
    -a | --array_size )
      ARRAY_SIZE="$2"
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
echo "Starting job: $JOB_NAME"
echo "Main config file: $CFG"
echo "Job array size: $ARRAY_SIZE"

# ************************************************* SLURM FUNCS **************************************************
make_creation_job_script() {
    # arguments
    #   conf: configuration file
    #   out : output path for generated bash script
    conf=$1
    out=$2
    # fill template
    readarray -t -d '\n\t' template << TTT
    #SBATCH -n 2
    #SBATCH --time=1:30:00
    #SBATCH --mem-per-cpu=2000
    #SBATCH --tmp=4000
    #SBATCH --job-name=$JOB_NAME
    #SBATCH --output=create_dataset_$JOB_NAME.stdout
    #SBATCH --error=create_dataset_$JOB_NAME.stderr

    module load xyz/123
    python /scratch/ewalt/pdm/rs-uncertainty/src/scripts/create_projects.py \
        --cfg $conf
TTT

    printf "%s\n" "${template[@]}" | awk '{$1=$1};1' > $out
    chmod +x $out
}

# make_aggregation_job_script() {
    # fill template
#         readarray -t -d '\n\t' template << TTT
#         #SBATCH -n 2
#         #SBATCH --time=1:30:00
#         #SBATCH --mem-per-cpu=2000
#         #SBATCH --tmp=4000
#         #SBATCH --job-name=$JOB_NAME
#         #SBATCH --output=create_dataset_$JOB_NAME.stdout
#         #SBATCH --error=create_dataset_$JOB_NAME.stderr

#         module load xyz/123
#         python /scratch/ewalt/pdm/rs-uncertainty/src/scripts/create_dataset.py \
#             --aggregate \
#             --cfg cfg
# TTT
# }


# **************************************************** LOGIC ****************************************************
# *(1)* Configure 
read -r -a sub_cfgs <<< `python /scratch/ewalt/pdm/rs-uncertainty/src/scripts/create_dataset.py --configure --cfg $CFG --num_jobs $ARRAY_SIZE`
# i=0
# for sub in "${sub_cfgs[@]}"
# do
#     ((i+=1))
#     echo "[$i] $sub"
# done
tmp=$(basename ${sub_cfgs[0]})
tmp=$(dirname ${sub_cfgs[0]})/`echo $tmp | sed 's/[0-9]/*/g'`
echo "Saved sub config files: $tmp"

# *(2)* Generate creation scripts
job_ids=()
for sub in "${sub_cfgs[@]}"
do
    # get sub id from config file name
    out=${sub:0:(-5)}.sh
    # generate script
    make_creation_job_script $sub $out
    # submit
    # job_id=`sbatch $out`
    job_id=`./fake_submit.sh`
    job_ids+=(${job_id##* })
done
tmp=$(basename ${sub_cfgs[0]:0:(-5)}.sh)
tmp=$(dirname ${sub_cfgs[0]})/`echo $tmp | sed 's/[0-9]/*/g'`
echo "Saved and submitted slurm jobs: $tmp"
echo "Slurm job ids: ${job_ids[@]}" 

# *(3)* Generate aggregation script
# make_aggregation_job_script ${job_ids[@]} ${sub_cfgs[@]}