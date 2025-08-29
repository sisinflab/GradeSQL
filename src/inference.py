import json
import math

import torch
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

from config.config import *
from test_time_scaling.test_time_scaling import *
from utils.sql_utilities import parse_response
from utils.n_decreasing import n_decreasing



if __name__ == '__main__':
    """
    Taken and adapted from the OmniSQL inference script.
    Runs SQL query generation experiments with configurable model, dataset, and inference parameters.


    This script:
    1. Loads configuration settings (model paths, dataset info, inference parameters).
    2. Generates N candidate SQL queries for each input example using a specified LLM.
    3. Processes model outputs into raw responses and extracted SQL strings.
    4. Saves predictions to a cache directory to avoid redundant computation.
    5. Optionally applies N-decreasing post-processing.

    The script supports batched generation for increasing speed and seed variation for diversity.

    Key Behaviors:
        - If cached predictions already exist, skips generation and optionally runs `n_decreasing`.
        - Uses `parse_response` to extract SQL code blocks from model outputs.
        - Stores both the raw responses and the parsed SQLs in the output JSON.
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
    NUM_SAMPLE_PER_BATCH = min(config['inference']['num_samples_per_batch'], N)
    
    print("=" * 50)
    print(" " * 10 + "Experiment Settings" + " " * 10)
    print("=" * 50)
    print(f"{'Model Name':<25}: {config['inference']['model_name']}")
    print(f"{'Max Tokens':<25}: {MAX_TOKENS}")
    print(f"{'N':<25}: {N}")
    print(f"{'N Decreasing':<25}: {N_DECREASING}")
    print(f"{'N Step':<25}: {N_STEP}")
    print(f"{'Seed':<25}: {SEED}")
    print(f"{'Temperature':<25}: {TEMPERATURE}")
    print(f"{'Start Query':<25}: {START_QUERY}")
    print(f"{'Offset Start Query':<25}: {OFFSET_START_QUERY}")
    print(f"{'Test Time Strategy':<25}: {TEST_TIME_STRATEGY}")
    print(f"{'Dataset Type':<25}: {TYPE_DATASET}")
    print(f"{'Spider or Bird':<25}: {dataset_name}")
    print("=" * 50)

    PRED_DIR = f"predictions_{dataset_name}_{TYPE_DATASET}_{N}_{SEED}_{TEMPERATURE}_{START_QUERY}_{MODEL_NAME}"
    PRED_DIR = conf.construct_path("../../cache/" + PRED_DIR)
    os.makedirs(PRED_DIR, exist_ok=True)
    output_file = PRED_DIR +  "/predict.json"

    input_dataset = json.load(open(DATASET_PATH))
    input_dataset = input_dataset[START_QUERY:START_QUERY+OFFSET_START_QUERY]

    if not os.path.exists(output_file):
        torch.manual_seed(SEED)
        tokenizer = AutoTokenizer.from_pretrained(PATH_MODEL_PRETRAINED, trust_remote_code=True)
        
        if "OmniSQL-" in MODEL_NAME:
            stop_token_ids = [151645]

        llm = LLM(
            model = PATH_MODEL_PRETRAINED,
            dtype = "bfloat16", 
            tensor_parallel_size = torch.cuda.device_count(),
            max_model_len = 8192,
            gpu_memory_utilization = 0.92,
            swap_space = 42,
            enforce_eager = True,
            disable_custom_all_reduce = True,
            trust_remote_code = True
        )
        
        chat_prompts = [tokenizer.apply_chat_template(
            [{"role": "user", "content": data["input_seq"]}],
            add_generation_prompt = True, tokenize = False
        ) for data in input_dataset]

        # Calculate number of batches
        num_batches = math.ceil(N / NUM_SAMPLE_PER_BATCH)
        all_results = []

        # Process each batch
        total_samples = 0
        for batch_idx in range(num_batches):
            
            # Calculate samples for this batch
            remaining_samples = N - total_samples
            current_batch_size = min(NUM_SAMPLE_PER_BATCH, remaining_samples)
            
            # Increment seed for diversity
            batch_seed = SEED + batch_idx

            sampling_params = SamplingParams(
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
                top_k=50,
                top_p=0.95,
                n=current_batch_size,
                stop_token_ids=stop_token_ids,
            )

            print(f"[Progress]: Generating batch {batch_idx + 1}/{num_batches} with {current_batch_size} samples, seed={batch_seed}")

            # Generate outputs for the batch
            outputs = llm.generate(chat_prompts, sampling_params)
            
            # Process outputs
            batch_results = []
            for data, output in zip(input_dataset, outputs):
                responses = [o.text for o in output.outputs]
                sqls = [parse_response(response) for response in responses]
                data_copy = data.copy()
                data_copy["responses"] = responses
                data_copy["pred_sqls"] = sqls
                batch_results.append(data_copy)
            
            # Append batch results to all results
            all_results.append(batch_results)
            total_samples += current_batch_size

        # Combine results: Merge responses and pred_sqls from all batches for each input
        final_results = []
        for idx in range(len(input_dataset)):
            combined_data = input_dataset[idx].copy()
            combined_responses = []
            combined_sqls = []
            for batch_results in all_results:
                combined_responses.extend(batch_results[idx]["responses"])
                combined_sqls.extend(batch_results[idx]["pred_sqls"])
            combined_data["responses"] = combined_responses[:N]
            combined_data["pred_sqls"] = combined_sqls[:N]
            final_results.append(combined_data)


        with open(output_file, "w", encoding = "utf-8") as f:
            f.write(json.dumps(final_results, indent = 2, ensure_ascii = False))
    else:
        print("[Cached]: Predictions already generated.")
        with open(output_file, "r", encoding="utf-8") as f:
            results = json.load(f)
        
        if N_DECREASING == True:
            n_decreasing()