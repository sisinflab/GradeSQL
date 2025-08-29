#!/bin/bash

# -----------------------------------------------------------------------------
# Script to generate multiple YAML config files for experiments by replacing
# placeholders in a template YAML file with calculated start_query and offset.
#
# For each config file:
#   - start_query is set to the starting index of the batch of queries
#   - offset is set to the fixed number of queries per batch
#
# The template file is located at: ../../recipes/config_template.yaml
# Output files are saved as: ../../recipes/1.yaml, 2.yaml, ..., NUM_FILES.yaml
# -----------------------------------------------------------------------------


OFFSET=38
NUM_FILES=40
TEMPLATE=$(cat ../../recipes/config_template.yaml)


for ((i=1; i<=NUM_FILES; i++))
do
 if [ $i -eq 1 ]; then
   START_QUERY=0
 else
   START_QUERY=$(( (i-1) * OFFSET ))
 fi

 OUTPUT=$(echo "$TEMPLATE" | sed "s/START_QUERY_VALUE/$START_QUERY/" | sed "s/OFFSET_VALUE/$OFFSET/")
 echo "$OUTPUT" > "../../recipes/${i}.yaml"
 echo "[Success]: Created ${i}.yaml with start_query = $START_QUERY and offset = $OFFSET"
done