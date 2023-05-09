# get config
CONFIG_FILE=$1
num_projects=`python -c "import yaml; f = open('${CONFIG_FILE}', 'r'); print(len(yaml.safe_load(f))['projects'])-1); f.close()"`

# Create multiple configs
paths=()
for i in {0..$num_projects}
do
    path=`python -c "import yaml, os, sys; f = open('${CONFIG_FILE}', 'r'); cfg = yaml.safe_load(f); f.close(); cfg['projects'] = cfg['projects'][${i}]; path = f'tmp/cp_cfg_{idx}.yaml'; with open(path, "w"): yaml.dump(cfg, path); print(path)"`
    echo "new temporary config $path"
    paths+=($path)
done

# submit creation jobs

# submit aggregation job

# delete tmp config
for path in ${paths[@]}
do
    echo "deleting $path"
    rm $path
done