CONFIG_FILE=$1
PROCESS_INDEX=$2

# parse log name
log_name=$(echo $(basename $CONFIG_FILE) | cut -d "." -f 1)
log_f=/cluster/work/igp_psr/elwalt/logs/train/{$log_name}_{$PROCESS_INDEX}.log
mkdir -p $(dirname $log_f)
touch log_f

# define slurm options
options="-n 2 --mem-per-cpu=12000 --time=18:00:00 --job-name=train --output=$log_f --error=$log_f --gpus=1 --gres=gpumem:20g"

# lauch
echo "Submitting training job"
retvalue=($(sbatch $options /cluster/work/igp_psr/elwalt/pdm/rs-uncertainty/run/slurm/train.sh $CONFIG_FILE))
echo "${retvalue[@]}"
echo "Done."
