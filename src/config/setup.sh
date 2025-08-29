#!/bin/bash

# 
# Environment Setup and Virtual Environment Management Script for GradeSQL Project
#
# This script prepares the environment to run experiments and utilities for the GradeSQL project.
#
# Features:
# - Sets up project-related environment variables including paths to scripts, dataset adapters,
#   Hugging Face cache locations, and Python paths.
# - Defines experiment name from the first script argument or defaults to "config_template".
# - Sources the user's bashrc for environment consistency.
# - Loads necessary modules (CUDA 12.1 and Python 3.11) for HPC environments.
# - Checks if virtual environments for base and vLLM setups exist:
#     - If missing, creates them and installs required dependencies from requirements files.
#     - If present, activates the base virtual environment.
# - Two virtual environments are present because the generation of CoTs uses the same dependecies as specified in OmniSQL github repository.
#   The other virtual enviroment is the one used for Test Time Scaling module, and ORM finetuning.
# Set the experiment name, either from the environment or passed as an argument


PROJECT_DIR=/your/absolute/path/to/GradeSQL
export EXP_NAME="config_template"  # Default value if no argument is passed
export PYTHONPATH=$PROJECT_DIR/src:$PYTHONPATH
export HF_HOME="$PROJECT_DIR/../../hf_cache"
export HUGGINGFACE_HUB_CACHE="$PROJECT_DIR/../../hf_cache"
VENV_DIR_BASE=$PROJECT_DIR/../../venv_base
VENV_DIR_VLLM=$PROJECT_DIR/../../venv_vllm
export PYTHONUNBUFFERED=1



# Paths to the various scripts for inference, evaluation, and experiment running
export INFERENCE=$PROJECT_DIR"/src/inference.py"
export CHOOSE=$PROJECT_DIR"/src/choose.py"
export EVALUATION="sh "$PROJECT_DIR"/src/evaluation/run_evaluation.sh"
export RUN_EXPERIMENT="sh "$PROJECT_DIR"/src/scripts/run_experiment.sh"
export RUN_BATCH_EXPERIMENTS="sh "$PROJECT_DIR"/src/scripts/run_batch_experiments.sh"
export CLEAN="rm -rf cache results"
source ~/.bashrc


export EXP_NAME=$1

REQUIREMENTS_BASE=$PROJECT_DIR/requirements_base.txt
REQUIREMENTS_VLLM=$PROJECT_DIR/requirements_vllm.txt


if [ ! -d "$VENV_DIR_BASE" ]; then
    # Load necessary modules for the experiment for HPC enviroment. If you're not in cluster enviroment, they will fail

    module load cuda/12.1
    module load python/3.11

    # If the virtual environment doesn't exist, create and set it up
    python3 -m venv $VENV_DIR_BASE
    source $VENV_DIR_BASE/bin/activate
    pip3 install --upgrade pip
    pip3 install --no-cache-dir -r $REQUIREMENTS_BASE

    deactivate
    module purge
    module load cuda/12.1
    module load python/3.11
    python3 -m venv $VENV_DIR_VLLM
    source $VENV_DIR_VLLM/bin/activate
    pip3 install --upgrade pip
    pip3 install --no-cache-dir -r $REQUIREMENTS_VLLM

    deactivate
    module purge
    module load cuda/12.1
    module load python/3.11
    source $VENV_DIR_BASE/bin/activate
else
    module load cuda/12.1
    module load python/3.11

    # If the virtual environment already exists, just activate it
    source $VENV_DIR_BASE/bin/activate
fi