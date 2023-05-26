# get project root
MACHINE=$1
if [[ "$MACHINE" == "--euler" ]]  || [[ "$MACHINE" == "--local" ]]; then
    echo "Running on: $MACHINE"
    job_ids=()
else
    echo "invalid machine"; exit 1
fi
if [[ $MACHINE == "--euler" ]]; then
    options="-n 1 --mem-per-cpu=64000 --time=4:00:00 --job-name=predict --gpus=1 --gres=gpumem:12g"
    root="/cluster/work/igp_psr/elwalt/pdm/rs-uncertainty"
else 
    root="/scratch/ewalt/pdm/rs-uncertainty"
fi

# get config
mapfile -t configTriplets < <(python ${root}/src/scripts/configure_cloud_experiment.py --create ${root}/config/create_dataset/cloud_exp_template.yaml --predict ${root}/config/predict_testset/cloud_exp_template.yaml --eval ${root}/config/predict_testset/cloud_exp_template.yaml)

# submit pipeline job
for (( i=0; i<${#configTriplets[@]}; i++ ));
do
    if [[ $MACHINE == "--euler" ]]; then 
        echo "Submitting pipeline job for: ${configTriplets[$i]}"
        readarray -td " " configTriplet <<< "${configTriplets[$i]}"
        # get log files
        log_name=$(echo $(basename $CONFIG_FILE) | cut -d "." -f 1)
        # create
        log_f=/cluster/work/igp_psr/elwalt/logs/pipeline/$log_name.log
        mkdir -p $(dirname $log_f)
        touch log_f
        # make options
        pipelineOptions="${options} --output=${log_f} --error=${log_f}"
        echo "Submitting job array ..."
        retvalue=($(sbatch $options /cluster/work/igp_psr/elwalt/pdm/rs-uncertainty/run/slurm/pipeline.sh ${configTriplets[$i]}))
        echo "${retvalue[@]}"
        job_ids+=(${retvalue[-1]})
    else
        set -e 
        readarray -td " " configTriplet <<< "${configTriplets[$i]}"
        echo "***** creating dataset *****"
        python ${root}/src/scripts/create_dataset.py --preprocess --cfg ${configTriplet[0]}
        echo "***** predicting testset *****"
        python ${root}/src/scripts/predict_testset.py --cfg ${configTriplet[1]}
        echo "***** evaluating testset *****"
        python ${root}/src/scripts/evaluate_testset.py --cfg ${configTriplet[2]}
        set +e
    fi 
done

if [[ $MACHINE == "--euler" ]]; then
    echo "Job ids: ${job_ids}"
fi
