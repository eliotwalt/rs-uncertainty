PYVERSION=3.7.4
module load python/$PYVERSION
pip install -q -r requirements.txt --user

wandb login

