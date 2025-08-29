#!/bin/bash

# -----------------------------------------------------------------------------
# Batch Job Submission Script for Inference Recipes
#
# This script:
#   1. Iterates over a sequence of recipes indices from `start_recipe` to `end_recipe`. 
#      So files have to be named 1.yaml, 2.yaml, etc.
#   2. For each recipe index, submits a Slurm batch job using `inference_sbatch.sh`.
#   3. Passes the current recipe index as an argument to the batch script.
# -----------------------------------------------------------------------------


start_recipe=1
end_recipe=40
step=1

for i in $(seq $start_recipe $step $end_recipe); do
    echo "Submitting job for recipe $i"
    sbatch src/scripts/inference_sbatch.sh "$i"
done