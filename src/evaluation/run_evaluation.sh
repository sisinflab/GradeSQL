#!/bin/bash

# ============================================
# Script to run evaluation of predicted SQL queries
# for a given experiment name, based on its YAML config.
# Usage:
#   ./run_evaluation.sh <experiment_name>
# ============================================



PROJECT_DIR=$(python3 src/config/get_project_dir.py)
cd $PROJECT_DIR/src/evaluation



exp_name=$1
if [ -z "$exp_name" ]; then
    echo "[Error] Please provide the experiment name as the first argument."
    exit 1
fi



yaml_file="../../recipes/$exp_name.yaml"
if [ ! -f "$yaml_file" ]; then
    echo "[Error] YAML file not found at $yaml_file"
    exit 1
fi



offset_start_query=$(python3 -c "import yaml; print(yaml.safe_load(open('$yaml_file'))['inference']['offset_start_query'])")
if [ -z "$offset_start_query" ]; then
    echo "[Error]: 'offset_start_query' not found in $yaml_file"
    exit 1
fi
echo "[Evaluation]: Number of queries: $offset_start_query"



timeout_sql=$(python3 -c "import yaml; print(yaml.safe_load(open('$yaml_file'))['inference']['timeout_sql'])")
if [ -z "$timeout_sql" ]; then
    echo "[Error]: 'timeout_sql' not found in $yaml_file"
    exit 1
fi



dataset_name=$(python3 -c "import yaml; print(yaml.safe_load(open('$yaml_file'))['dataset']['dataset_name'])")
if [ -z "$dataset_name" ]; then
    echo "[Error]: 'dataset_name' not found in $yaml_file"
    exit 1
fi



is_train_dev_test=$(python3 -c "import yaml; print(yaml.safe_load(open('$yaml_file'))['dataset']['is_train_dev_test'])")
if [ -z "$is_train_dev_test" ]; then
    echo "[Error]: 'is_train_dev_test' not found in $yaml_file"
    exit 1
fi



db_root_path=$(python3 -c "import yaml; print(yaml.safe_load(open('$yaml_file'))['dataset']['db_sqlite'])")
if [ -z "$db_root_path" ]; then
    echo "[Error]: 'db_root_path' not found in $yaml_file"
    exit 1
fi



ground_truth_path=$(python3 -c "import yaml; print(yaml.safe_load(open('$yaml_file'))['dataset']['ground_truth_path'])")
if [ -z "$ground_truth_path" ]; then
    echo "[Error]: 'ground_truth_path' not found in $yaml_file"
    exit 1
fi



dataset_path=$(python3 -c "import yaml; print(yaml.safe_load(open('$yaml_file'))['dataset']['dataset_path'])")
if [ -z "$dataset_path" ]; then
    echo "[Error]: 'dataset_path' not found in $yaml_file"
    exit 1
fi



tables=$(python3 -c "import yaml; print(yaml.safe_load(open('$yaml_file'))['dataset']['tables_path'])")
if [ -z "$tables" ]; then
    echo "[Error]: 'tables_path' not found in $yaml_file"
    exit 1
fi


if [ $dataset_name = "bird" ]; then

    data_mode=$is_train_dev_test
    diff_json_path=$dataset_path
    predicted_sql_path_kg="../../results/$exp_name/"
    num_cpus=16
    mode_gt='gt'
    mode_predict='gpt'

    echo '''Starting to compare with knowledge for EX'''
    python3 -u ./evaluation_ex_bird.py --db_root_path ${db_root_path} --predicted_sql_path ${predicted_sql_path_kg} --data_mode ${data_mode} \
    --ground_truth_path ${ground_truth_path} --num_cpus ${num_cpus} --mode_gt ${mode_gt} --mode_predict ${mode_predict} \
    --diff_json_path ${diff_json_path} --meta_time_out ${timeout_sql} --num_queries ${offset_start_query}
fi



test_time_strategy=$(python3 -c "import yaml; print(yaml.safe_load(open('$yaml_file'))['inference']['test_time_strategy'])")
if [ -z "$test_time_strategy" ]; then
    echo "[Error]: 'test_time_strategy' not found in $yaml_file"
    exit 1
fi



predicted_sql_path_kg="../../results/$exp_name/predict.json"
if [ $test_time_strategy = "pass@k" ]; then
    python3 ./evaluation_omni.py --pred ${predicted_sql_path_kg} --gold ${dataset_path} --db_path ${db_root_path} --mode 'pass@k'
else
    python3 ./evaluation_omni.py --pred ${predicted_sql_path_kg} --gold ${dataset_path} --db_path ${db_root_path} --mode 'greedy_search'
fi