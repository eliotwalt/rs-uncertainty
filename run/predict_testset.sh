# input config file
CONFIG_FILE=$1

# parse job name 
job_name=$(echo $(basename $CONFIG_FILE) | cut -d "." -f 1)
log_f=/cluster/work/igp_psr/elwalt/logs/predict/$job_name.log
mkdir -p $(dirname $log_f)
touch log_f

# define slurm options
options="-n 1 --mem-per-cpu=64000 --time=4:00:00 --job-name=$job_name --gpus=1 --gres=gpumem:12g --output=$log_f --error=$log_f"

# lauch
echo "Submitting job '$job_name'"
retvalue=($(sbatch $options /cluster/work/igp_psr/elwalt/pdm/rs-uncertainty/run/slurm/predict.sh $CFG))
echo "${retvalue[@]}"
echo "Done."
