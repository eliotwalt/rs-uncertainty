# get project root
MACHINE=$1
if [[ "$MACHINE" == "--euler" ]]  || [[ "$MACHINE" == "--local" ]]; then
    echo "Running on: $MACHINE"
    job_ids=()
else
    echo "invalid machine"; exit 1
fi
if [[ $MACHINE == "--euler" ]]; then
    echo "Setting up env"
    PYVERSION=3.7.4
    GDALVESION=3.4.3
    module load python/$PYVERSION gdal/$GDALVERSION
    pip install gdal==`gdal-config --version` --user # do not put it in requirements, the version must match local install!
    pip install -r requirements.txt --user
    options="-n 1 --mem-per-cpu=24000 --time=45:00 --job-name=pipeline_one_img_ds --gpus=1 --gres=gpumem:4g"
    root="/cluster/work/igp_psr/elwalt/pdm/rs-uncertainty"
else 
    root="/scratch/ewalt/pdm/rs-uncertainty"
fi

# get config
echo "Running autoconfig..."
createConfig=$2
predictConfig=$3
evalConfig=$4
if [[ $MACHINE == "--euler" ]]; then
    mapfile -t configTriplets < <(python ${root}/src/scripts/configure_one_image_dataset_experiment.py --create ${createConfig} --predict ${predictConfig} --eval ${evalConfig} --tmp_dir ${root}/tmp)
else 
    mapfile -t configTriplets < <(python ${root}/src/scripts/configure_one_image_dataset_experiment.py --create ${createConfig} --predict ${predictConfig} --eval ${evalConfig})
fi
experiment_name=`python -c "from pathlib import Path; print(Path('${createConfig}').stem)"`

# submit pipeline job
if [[ $MACHINE == "--euler" ]]; then 
    for (( i=0; i<${#configTriplets[@]}; i++ ));
    do
        read -a configTriplet <<< "${configTriplets[$i]}"
        echo "Submitting pipeline job for: ${configTriplet[@]}"
        # get log files
        log_suffix=`python -c "print('_'.join('${configTriplet[0]}'.split('/')[-2].split('_')[1:]))"`
        # create
        log_f=/cluster/work/igp_psr/elwalt/logs/pipeline/${experiment_name}_${log_suffix}.log
        mkdir -p $(dirname $log_f)
        touch $log_f
        echo "log file is: ${log_f}"
        # make options
        pipelineOptions="${options} --output=${log_f} --error=${log_f}"
        retvalue=($(sbatch $options /cluster/work/igp_psr/elwalt/pdm/rs-uncertainty/run/slurm/pipeline.sh ${configTriplets[$i]}))
        echo "${retvalue[@]}"
        job_ids+=(${retvalue[-1]})
    done
else
    run_pipeline() {
        set -e 
        echo "***** creating dataset *****"
        python ${root}/src/scripts/create_dataset.py --preprocess --cfg $1
        echo "***** predicting testset *****"
        python ${root}/src/scripts/predict_testset.py --cfg $2
        echo "***** evaluating testset *****"
        python ${root}/src/scripts/evaluate_testset.py --cfg $3
        set +e
    }
    # for (( i=0; i<${#configTriplets[@]}; i+=4 ));
    for (( i=0; i<${#configTriplets[@]}; i++ ));
    do
        # for (( j=i; j<i+4; j+=4 ));
        # do
        #     (run_pipeline ${configTriplets[$j]}) &
        # done
        # wait
        run_pipeline ${configTriplets[$i]}
    done
fi 

if [[ $MACHINE == "--euler" ]]; then
    echo "Job ids: ${job_ids}"
fi
