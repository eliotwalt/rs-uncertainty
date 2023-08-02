echo "Launching pipeline 10"
./run/launch_one_image_datasets_experiment.sh --euler \
    config/create_dataset/empirical_cloud_threshold_exp_10_gee_template.yaml \
    config/predict_testset/empirical_cloud_threshold_exp_10_gee_template.yaml \
    config/evaluate_testset/empirical_cloud_threshold_exp_10_gee_template.yaml
echo "Launching pipeline 30"
./run/launch_one_image_datasets_experiment.sh --euler \
    config/create_dataset/empirical_cloud_threshold_exp_30_gee_template.yaml \
    config/predict_testset/empirical_cloud_threshold_exp_30_gee_template.yaml \
    config/evaluate_testset/empirical_cloud_threshold_exp_30_gee_template.yaml
echo "Launching pipeline 70"
./run/launch_one_image_datasets_experiment.sh --euler \
    config/create_dataset/empirical_cloud_threshold_exp_70_gee_template.yaml \
    config/predict_testset/empirical_cloud_threshold_exp_70_gee_template.yaml \
    config/evaluate_testset/empirical_cloud_threshold_exp_70_gee_template.yaml