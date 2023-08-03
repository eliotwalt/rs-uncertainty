M=$1
for i in $(seq 1 $M);
do
    ./run/train.sh ./config/train/empirical_cloud_threshold_exp_10.yaml $i
    ./run/train.sh ./config/train/empirical_cloud_threshold_exp_30.yaml $i
    ./run/train.sh ./config/train/empirical_cloud_threshold_exp_70.yaml $i
done
