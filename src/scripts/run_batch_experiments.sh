#!/bin/bash

# -----------------------------------------------------------------------------
# Batch Experiment Runner
#
# Description:
#   Iterates through all recipe configuration files in the project's `recipes`
#   directory and runs experiments for each one that:
#     1. Is not named `config_template`.
#     2. Has not already produced complete results in the `results` directory.
# -----------------------------------------------------------------------------


export EXP_NAME=config_template
PROJECT_DIR=$(python3 src/config/get_project_dir.py)
DIR=$PROJECT_DIR"/recipes"

for file in "$DIR"/*; do
    if [ -f "$file" ]; then
        filename=$(basename "$file" .${file##*.})

        if [ "$filename" = "config_template" ]; then
            echo "[Progress]: Skipping $file (filename is config_template)"
            continue
        fi
        
        if [ -f "$PROJECT_DIR/results/$filename/results.txt" ] && [ -f "$PROJECT_DIR/results/$filename/predict.json" ] && [ -f "$PROJECT_DIR/results/$filename/logger.log" ] &&  [ -f "$PROJECT_DIR/results/$filename/config.yaml" ]; then
            echo "[Progress]: Experiment $file already done. Skipping it."
            continue
        fi

        eval "$RUN_EXPERIMENT $filename"
    fi
done