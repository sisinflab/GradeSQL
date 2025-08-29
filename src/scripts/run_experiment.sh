#!/bin/bash

# -----------------------------------------------------------------------------
# Experiment Runner Script
#
# Description:
#   This script automates running an experiment pipeline consisting of:
#     1. Inference stage (`$INFERENCE`).
#     2. SQL selection (`$CHOOSE`).
#     3. Evaluation stage (`$EVALUATION`).
#
#   It manages environment activation/deactivation between stages,
#   and stores experiment results in results folder.
# -----------------------------------------------------------------------------


export EXP_NAME="config_template"
PROJECT_DIR=$(python3 src/config/get_project_dir.py)
VENV_DIR_BASE=$PROJECT_DIR/../../venv_base
VENV_DIR_VLLM=$PROJECT_DIR/../../venv_vllm


cat <<'EOF'
   ____               _      ____   ___  _     
  / ___|_ __ __ _  __| | ___/ ___| / _ \| |    
 | |  _| '__/ _` |/ _` |/ _ \___ \| | | | |    
 | |_| | | | (_| | (_| |  __/___) | |_| | |___ 
  \____|_|  \__,_|\__,_|\___|____/ \__\_\_____|
                                              
EOF


if [ $# -eq 0 ]; then
    echo "Please provide the name of the experiment you're running."
else
    echo "[Progress]: Running experiment $1..."

    while true; do
        export EXP_NAME=$1

        module purge
        module load cuda/12.1
        module load python/3.11
        source $VENV_DIR_VLLM/bin/activate

        eval "python3 $INFERENCE"

        deactivate
        module purge
        module load cuda/12.1
        module load python/3.11
        source $VENV_DIR_BASE/bin/activate
        
        eval "python3 $CHOOSE"
        
        exit_code=$?
        
        if [ $exit_code -eq 0 ]; then
            echo "[Progress]: Script completed successfully with exit code 0. Exiting loop."
            break
        else
            echo "[Progress]: Script failed with exit code $exit_code. Some problem has occured. Exiting loop..."
            break
        fi
    done

    mkdir -p $PROJECT_DIR/results/$1
    eval "$EVALUATION $1" > $PROJECT_DIR/results/$1/results.txt 2>&1
    cp $PROJECT_DIR/recipes/$1.yaml $PROJECT_DIR/results/$1/config.yaml
fi