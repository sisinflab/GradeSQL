#!/bin/bash

# -----------------------------------------------------------------------------
# Script to generate multiple YAML config files from a template by replacing
# the placeholder N_MAX with decreasing values starting from START_N, stepping
# down by STEP each time until 1.
# -----------------------------------------------------------------------------


START_N=32
STEP=1
TEMPLATE=$(cat ../../recipes/config_template.yaml)

NUM_FILES=$(( (START_N + STEP - 1) / STEP ))

for ((i=1; i<=NUM_FILES; i++))
do
  CURRENT_N=$(( START_N - (i-1) * STEP ))
  OUTPUT=$(echo "$TEMPLATE" | sed "s/N_MAX/$CURRENT_N/")
  echo "$OUTPUT" > "../../recipes/${CURRENT_N}.yaml"
  echo "[Success]: Created ${CURRENT_N}.yaml with N = $CURRENT_N"
done