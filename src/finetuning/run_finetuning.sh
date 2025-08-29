#!/bin/bash


# ------------------------------------------------------------------------------
# SLURM batch script for launching finetuning jobs with experiment recipes.
# 
# Description:
#   - Checks that a finetuning recipe YAML file exists.
#   - Extracts the 'finetuning_version' from the YAML.
#   - Based on the finetuning version, runs the corresponding finetuning and evaluation scripts.
#   - Supports finetuning versions "v1" and "v2".
#   - Sets environment variables for NCCL and OpenMP thread usage.
#   - Sends email notifications on job end or failure.
# ------------------------------------------------------------------------------



#SBATCH --job-name=finetuning
#SBATCH --partition=enter_your_partition_here
#SBATCH --qos=normal
#SBATCH --mem=100GB
#SBATCH --account=enter_your_account_here
#SBATCH --ntasks=1
#SBATCH -N 1
#SBATCH --time=3-23:59:59
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
#SBATCH --output=finetuning_%j.out
#SBATCH --error=finetuning_%j.err
#SBATCH --mail-type=END,FAIL    
#SBATCH --mail-user=enter_your_email

if [ -z "$1" ]; then
    echo "[Error]: Provide as parameter the name of the finetuning recipe."
    exit 1
fi

export NCCL_SOCKET_IFNAME=ib0
export OMP_NUM_THREADS=4

cd ../../
source src/config/setup.sh $1
cd src/finetuning
export EXP_NAME=$1
PROJECT_DIR=$(python3 ../../src/config/get_project_dir.py)

yaml_file="../../recipes/$EXP_NAME.yaml"
if [ ! -f "$yaml_file" ]; then
    echo "[Error] YAML file not found at $yaml_file"
    exit 1
fi

finetuning_version=$(python3 -c "import yaml; print(yaml.safe_load(open('$yaml_file'))['orm']['finetuning_version'])")
if [ -z "$finetuning_version" ]; then
    echo "[Error]: 'finetuning_version' not found in $yaml_file"
    exit 1
fi
echo "[Finetuning]: Finetuning version: $finetuning_version"


if [ $finetuning_version = "v1" ]; then
    python finetuning.py
elif [ $finetuning_version = "v2" ]; then
    python finetuning_v2.py
fi

cd ../../
$RUN_EXPERIMENT $1