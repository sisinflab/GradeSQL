import json
import os

from config.config import Config



"""
Merges SQL predictions results from multiple nodes into a single `predict.json`.
Merges SQL choosen by test time scaling module from multiple nodes into a single `checkpoint.json`.

This script:
1. Loads configuration from `Config`.
2. Iterates over prediction and checkpoint output directories corresponding to multiple processing nodes.
3. Reads each node's `predict.json` and `checkpoint.json` file and appends its contents to a combined results list.
4. Saves the merged predictions and checkpoints into a new cache directory with `_merged` appended to its name.
"""


NUM_NODES = 40
NUM_QUERIES_PER_NODE = 38



conf = Config()
config = conf.get_config()
MODEL_NAME = config['inference']['model_name'].replace("/", "_")
DATASET_PATH = conf.construct_path(config['dataset']['dataset_path'])
N = config['inference']['n']
SEED = config['general']['seed']
TEMPERATURE = config['inference']['temperature']
TYPE_DATASET = config['dataset']['is_train_dev_test']
dataset_name = config['dataset']['dataset_name']
MAX_TOKENS = config['inference']['max_tokens']



new_results_pred = []
new_results_checkpoint = []

for i in range(NUM_NODES):
    PRED_DIR = f"predictions_{dataset_name}_{TYPE_DATASET}_{N}_{SEED}_{TEMPERATURE}_{i * NUM_QUERIES_PER_NODE}_{MODEL_NAME}"
    PRED_DIR = conf.construct_path(f"../../cache/{PRED_DIR}")
    output_file = os.path.join(PRED_DIR, "predict.json")
    with open(output_file, "r", encoding="utf-8") as f: 
        results = json.load(f)

    new_results_pred.extend(results)



    CHECK_DIR = f"checkpoint_{i+1}"
    CHECK_DIR = conf.construct_path(f"../../cache/{CHECK_DIR}")
    output_file = os.path.join(CHECK_DIR, "checkpoint.json")
    with open(output_file, "r", encoding="utf-8") as f: 
        results = json.load(f)
    results = results["selected_sqls"]
    new_results_checkpoint.extend(results)






PRED_DIR = f"predictions_{dataset_name}_{TYPE_DATASET}_{N}_{SEED}_{TEMPERATURE}_{0}_{MODEL_NAME}_merged"
PRED_DIR = conf.construct_path(f"../../cache/{PRED_DIR}")
os.makedirs(PRED_DIR, exist_ok=True)
output_file = os.path.join(PRED_DIR, "predict.json")


with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(new_results_pred, f, indent=4, ensure_ascii=False)



CHECK_DIR = f"checkpoint_1_merged"
CHECK_DIR = conf.construct_path(f"../../cache/{CHECK_DIR}")
os.makedirs(CHECK_DIR, exist_ok=True)
output_file = os.path.join(CHECK_DIR, "checkpoint.json")


new_results_checkpoint = {"selected_sqls": new_results_checkpoint,
                          "last_index": len(new_results_checkpoint)-1}

with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(new_results_checkpoint, f, indent=4, ensure_ascii=False)