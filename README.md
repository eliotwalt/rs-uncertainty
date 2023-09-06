# Uncertainty and Robustness in Remote Sensing for Environmental Monitoring

This repository contains the code for my Master's thesis written at the Photogrammetry and Remote Sensing Group at ETH Zürich. Most of the code is taken from [this repository](https://github.com/prs-eth/bayes-forest-structure), to which we added evaluation and plotting utilities. 

Note that the data folders are not included. The drive containing this was mounted at the root of the repository in a folder called `assets/`. Results and additional data can be found on Euler. 

## Code organization

* `./config` configuration files for different tasks
* `./gee` all things Google Earth Engine and Google Drive
  * `download_timeseries.py` is the main script used to download non-aggregated timeseries of Sentinel-2 images associated to a given ALS project
  * `gee_download.py` and `gdrive_handler.py` take care of downloading GEE images to a Google drive and locally automatically
  * This code requires to get `credentials.json` and `token.json` from the [Google Cloud API](https://cloud.google.com/apis/) after creating a GEE project and authorizing the use of Google drive API on your own account 

* `./notebooks` notebooks containing the plots of the main analysis
* `./run` bash scripts designed to launch specific jobs on Euler
  * Note that the concept of "one image dataset" is used throughout the code for compatibility with the original code organization. It consists in creating a dataset from a single image by changing a flat file structure into a nested file structure. see `src` for more details
* `./src` contains the core of the python code
  * `./src/scripts` contains the code for the main tasks (training, evaluating, etc.)
    * `configure_one_image_dataset_experiment.py` can be used to generate "one image datasets" configuration files from a triplet of create/predict/eval template config files (see `./config/{create_dataset,evaluate_dataset,predict_testset}/*_template.yaml`) using symbolic links to temporary directories.
  * `viz.py` contains the code for all multi-level plots shown in the analysis
  * `metrics.py` contains the online computation implementation of the RCU metrics