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

# submit pipeline job
for (( i=0; i<${#configTriplets[@]}; i++ ));
do
    read -a configTriplet <<< "${configTriplets[$i]}"
    if [[ $MACHINE == "--euler" ]]; then 
        echo "Submitting pipeline job for: ${configTriplet[@]}"
        # get log files
        log_name=$(echo $(basename ${configTriplet[0]}) | cut -d "." -f 1)
        # create
        log_f=/cluster/work/igp_psr/elwalt/logs/pipeline/$log_name.log
        mkdir -p $(dirname $log_f)
        touch log_f
        # make options
        pipelineOptions="${options} --output=${log_f} --error=${log_f}"
        retvalue=($(sbatch $options /cluster/work/igp_psr/elwalt/pdm/rs-uncertainty/run/slurm/pipeline.sh ${configTriplets[$i]}))
        echo "${retvalue[@]}"
        job_ids+=(${retvalue[-1]})
    else
        set -e 
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
