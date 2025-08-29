import json
from tqdm import tqdm

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel

from config.config import *
from test_time_scaling.test_time_scaling import *



"""
Runs the test-time SQL selection stage on pre-generated predictions.

This script:
1. Loads previously generated SQL predictions from a cached `predict.json`.
2. Optionally resumes from a saved checkpoint to avoid reprocessing completed examples.
3. Depending on the configured `test_time_strategy`, selects the best SQL query
   for each example using heuristics, execution results, majority voting, or
   ORM-based model scoring.
4. Saves the selected queries into:
   - An updated `predict.json` for further evaluation.
   - Format-specific output files (JSON for Bird, SQL for Spider).

Key Behaviors:
    - Supports checkpointing to continue from the last processed index.
    - Can load and prepare fine-tuned ORM models (`v1` or `v2`) for semantic reranking.
    - Uses `test_time_scaling()` to apply the chosen selection strategy.
    - Handles Bird and Spider datasets differently for output formatting.
"""




logger = setup_logging_erase()
conf = Config()
config = conf.get_config()

MODEL_NAME = config['inference']['model_name'].replace("/", "_")
PATH_MODEL_PRETRAINED = conf.construct_path(config['inference']['path_model_pretrained'])
MAX_TOKENS = config['inference']['max_tokens']
DATASET_PATH = conf.construct_path(config['dataset']['dataset_path'])
N = config['inference']['n']
N_DECREASING = config['inference']['n_decreasing']
N_STEP = config['inference']['n_step']
SEED = config['general']['seed']
TEMPERATURE = config['inference']['temperature']
START_QUERY = config['inference']['start_query']

OFFSET_START_QUERY = config['inference']['offset_start_query']
if config["general"]["debug"] == "true":
    OFFSET_START_QUERY = 3

TEST_TIME_STRATEGY = config['inference']['test_time_strategy']
TYPE_DATASET = config['dataset']['is_train_dev_test']
dataset_name = config['dataset']['dataset_name']

PRED_DIR = f"predictions_{dataset_name}_{TYPE_DATASET}_{N}_{SEED}_{TEMPERATURE}_{START_QUERY}_{MODEL_NAME}"
PRED_DIR = conf.construct_path("../../cache/" + PRED_DIR)
os.makedirs(PRED_DIR, exist_ok=True)
output_file = PRED_DIR +  "/predict.json"

input_dataset = json.load(open(DATASET_PATH))
input_dataset = input_dataset[START_QUERY:START_QUERY+OFFSET_START_QUERY]


print("[Progress]: Loading predictions generated...")
with open(output_file, "r", encoding="utf-8") as f:
    results = json.load(f)

print("[Progress]: Starting test time scaling module...")
CHECKPOINT_DIR = f"{conf.project_dir}/cache/checkpoint_{conf.exp_name}"
CHECKPOINT_FILE = os.path.join(CHECKPOINT_DIR, "checkpoint.json")
if not os.path.exists(CHECKPOINT_DIR):
    os.makedirs(CHECKPOINT_DIR)
start_index = 0
if os.path.exists(CHECKPOINT_FILE):
    with open(CHECKPOINT_FILE, "r") as f:
        checkpoint_data = json.load(f)
        start_index = checkpoint_data["last_index"] + 1
        selected_sqls = [checkpoint_data["selected_sqls"]]
        print(f"[Checkpoint]: Checkpoint found, starting from {start_index}")
else:
    selected_sqls = []



orm_model = None
orm_tokenizer = None

if TEST_TIME_STRATEGY == "best_of_n_orm":

    FINETUNING_VERSION = config["orm"]["finetuning_version"]
    BIRD_OR_SPIDER = config["dataset"]["dataset_name"]
    MODEL_NAME = config["orm"]["base_model_name"]
    ORM_OR_PRM = config["orm"]["orm_or_prm"]
    VERSION = config["orm"]["version_model_finetuned"]
    
    base_model_short_name = MODEL_NAME.split("/")[-1]
    MODEL_NAME_FINETUNED_ORM = f"{base_model_short_name}-{ORM_OR_PRM}-{BIRD_OR_SPIDER}-finetuned-{VERSION}"
    MODEL_NAME_FINETUNED_ORM = f"{conf.project_dir}/{MODEL_NAME_FINETUNED_ORM}"
    MODEL_NAME_BASE_ORM = config["orm"]["base_model_name"]
    NAME_TOKENIZER = MODEL_NAME_BASE_ORM.lower().replace('/', '_') + '_tokenizer'
    path_peft_model = f"{MODEL_NAME_FINETUNED_ORM}"

    if FINETUNING_VERSION == "v1":
        orm_tokenizer = AutoTokenizer.from_pretrained(f"{conf.project_dir}/../../hf_cache/{NAME_TOKENIZER}/", use_fast=True)
        if "32B" in MODEL_NAME_BASE_ORM:
            base_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME_BASE_ORM, torch_dtype="auto", device_map="auto")
        else:
            base_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME_BASE_ORM, torch_dtype=torch.float16, device_map={"": 0})
        
        path_peft_model = f"{MODEL_NAME_FINETUNED_ORM}"
        peft_model = PeftModel.from_pretrained(base_model, path_peft_model)
        orm_model = peft_model.merge_and_unload()
        del base_model 
        del peft_model  
   
        if "llama" in NAME_TOKENIZER:
            print("[Info]: [LLama model detected, setting PAD token as EOS token")
            orm_tokenizer.pad_token = orm_tokenizer.eos_token  # Use EOS token as padding for LLama models
            
    elif FINETUNING_VERSION == "v2":
        id2label = {1: "Yes", 0: "No"}
        label2id = {"Yes": 1, "No": 0}
        orm_model = AutoModelForSequenceClassification.from_pretrained(path_peft_model, num_labels=2, id2label=id2label, label2id=label2id,).to("cuda:0")
        orm_tokenizer = AutoTokenizer.from_pretrained(f"{conf.project_dir}/../../hf_cache/{NAME_TOKENIZER}/", use_fast=True)


    torch.cuda.empty_cache()


for i, entry in tqdm(enumerate(results), total=len(results), desc="Choosing the best query"):
    if i < start_index:
        continue

    if dataset_name == "bird":
        name_field = "SQL"
        evidence = entry["evidence"]
    else:
        name_field = "query"
        evidence = ""

    gold = None
    if TEST_TIME_STRATEGY == "orm_finetuning":
        gold = entry[name_field]
    
    selected_sql = test_time_scaling(entry["pred_sqls"], input_dataset[i]["db_id"], input_dataset[i]["question"]+evidence, strategy=TEST_TIME_STRATEGY, _id=i, orm_model=orm_model, orm_tokenizer=orm_tokenizer, question=input_dataset[i]["question"], gold=gold, schema=input_dataset[i]["schema"])
    selected_sqls.append(selected_sql)

    checkpoint_data = {
        "last_index": i,
        "selected_sqls": selected_sqls
    }

    with open(CHECKPOINT_FILE, "w") as f:
        json.dump(checkpoint_data, f, indent=4)



for result, selected_sql in zip(results, selected_sqls):

    if TEST_TIME_STRATEGY != "pass@k":
        result["pred_sqls"] = [selected_sql]
    else:
        result["pred_sqls"] = selected_sql



output_file = conf.project_dir + "/results/" + conf.get_exp_name()
os.makedirs(output_file, exist_ok=True)
output_file = output_file + "/predict.json"

with open(output_file, "w", encoding = "utf-8") as f:
    f.write(json.dumps(results, indent = 2, ensure_ascii = False))

if dataset_name == "bird":
    output_json = {}
    for i, result in enumerate(results):
        query = result["pred_sqls"][0]
        db_id = result["db_id"]
        output_json[str(i)] = f"{query}\t----- bird -----\t{db_id}"
    output_file = conf.project_dir + "/results/" + conf.get_exp_name() + f"/predict_{TYPE_DATASET}.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_json, f, indent=4, ensure_ascii=False)
else:
    output_lines = []
    for i, result in enumerate(results):
        query = result["pred_sqls"][0].replace("\n", " ")
        db_id = result["db_id"]
        output_lines.append(f"{query}")

    output_file = conf.project_dir + "/results/" + conf.get_exp_name() + f"/predict_{TYPE_DATASET}.sql"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        for line in output_lines:
            f.write(line.strip() + '\n')