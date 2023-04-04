import os, sys
root = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, root)
import rasterio
from metrics import RUQMetrics 

# 0. Global variables
PREDICTIONS_DIR = 
PKL_DIR =
GT_DIR =
EAST = ['346', '9', '341', '354', '415', '418', '416', '429', '439', '560', '472', '521', '498',
        '522', '564', '764', '781', '825', '796', '805', '827', '891', '835', '920', '959', '1023', '998',
        '527', '477', '542', '471']
WEST = ['528', '537', '792', '988', '769']
NORTH = ['819', '909', '896']
ALL = EAST + WEST + NORTH

# Get min/max predicted variances for online binning

# Iterate on projects
    # load test predictions
    # load test gt
    # load train predicitons
    # compute training_means and training_stds
    # standardize predicitions and gt
    # add project to online metrics

# Aggregate projects

# Save meetrics

def main():
    """
    - load standardization data
    - loop on projects, compute variance bounds online
    - init rcu
    - loop on projects
        - standardize
        - add project
        - get([project_id])
    - loop on regions
        - get(region)
    - save results (incl. histogram)
    """
    pass