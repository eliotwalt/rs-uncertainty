M=$1
for i in $(seq 1 $M);
do
    ./run/proto/train.sh ./config/train/cloudy_training.yaml $i
done
