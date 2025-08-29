import json
import os

from config.config import Config



def n_decreasing():

    """
    Iteratively reduces the number of predicted SQLs in stored results and saves
    new versions of the prediction files for each reduced count.

    This function reads a prediction file, decreases the number of SQL predictions
    (`pred_sqls`) in fixed steps (`n_step`) from the original `n` down to 1,
    and writes each reduced set into a separate directory.
    """
        
    conf = Config()
    config = conf.get_config()
    MODEL_NAME = config['inference']['model_name'].replace("/", "_")
    N = config['inference']['n']
    STEP = config['inference']['n_step']
    SEED = config['general']['seed']
    TEMPERATURE = config['inference']['temperature']
    TYPE_DATASET = config['dataset']['is_train_dev_test']
    START_QUERY = config['inference']['start_query']
    dataset_name = config['dataset']['dataset_name']
    PRED_DIR = f"predictions_{dataset_name}_{TYPE_DATASET}_{N}_{SEED}_{TEMPERATURE}_{START_QUERY}_{MODEL_NAME}"
    PRED_DIR = conf.construct_path(f"../../cache/{PRED_DIR}")

    output_file = os.path.join(PRED_DIR, "predict.json")
    with open(output_file, "r", encoding="utf-8") as f:
        results = json.load(f)


    for n_decreased in range(N - STEP, 0, -STEP):
        new_results = results.copy()
        for result, new_result in zip(results, new_results):
            new_result["pred_sqls"] = result["pred_sqls"][:n_decreased]

        NEW_PRED_DIR = f"predictions_{dataset_name}_{TYPE_DATASET}_{n_decreased}_{SEED}_{TEMPERATURE}_{START_QUERY}_{MODEL_NAME}"
        NEW_PRED_DIR = conf.construct_path(f"../../cache/{NEW_PRED_DIR}")
        os.makedirs(NEW_PRED_DIR, exist_ok=True)

        new_output_file = os.path.join(NEW_PRED_DIR, "predict.json")
        with open(new_output_file, "w", encoding="utf-8") as f:
            json.dump(new_results, f, indent=2, ensure_ascii=False)
