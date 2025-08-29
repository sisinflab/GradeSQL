#!/bin/bash

# -----------------------------------------------------------------------------
# SLURM Job Submission Script for Experiment Run
#
# Description:
#   Submits a single SLURM job to run a specified experiment using the
#   environment and configuration defined in `src/config/setup.sh`.
# -----------------------------------------------------------------------------



#SBATCH --job-name=inference
#SBATCH --partition=enter_your_partition_here 
#SBATCH --qos=normal
#SBATCH --mem=100GB
#SBATCH --account=enter_your_account_here
#SBATCH --ntasks=1
#SBATCH -N 1
#SBATCH --time=23:59:59
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:1
#SBATCH --output=test_%j.out
#SBATCH --error=test_%j.err
#SBATCH --mail-type=END,FAIL    
#SBATCH --mail-user=enter_your_email_here



source src/config/setup.sh $1
export EXP_NAME=$1
PROJECT_DIR=$(python3 src/config/get_project_dir.py)
$RUN_EXPERIMENT $1