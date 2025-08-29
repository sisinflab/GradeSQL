import os
import json
import random
import numpy as np
import torch

from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from transformers import set_seed as transformers_set_seed
from peft import LoraConfig, get_peft_model
from datasets import load_from_disk
from collections import Counter

from config.config import Config



# ------------------------------------------------------------------------------
# Python script for fine-tuning causal language models with LoRA on custom datasets.
#
# Features:
#   - Loads dataset from disk, applies prompt formatting.
#   - Uses Huggingface Transformers Trainer with LoRA adapter integration.
#   - Supports checkpoint resuming, multi-iteration fine-tuning, and validation split.
#   - Applies deterministic seeding for reproducibility.
#   - Supports HPC and local environment dataset paths and tokenizer loading.
#   - Saves training loss per iteration to JSON files.
#
# Configuration:
#   - Controlled via Config class in config/config.py
#   - Configurable parameters: seed, model names, finetuning iterations, batch sizes,
#     learning rates, LoRA parameters, etc.
# ------------------------------------------------------------------------------




conf = Config()
config = conf.get_config()
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
MACHINE = conf.normal_or_hpc
VALIDATION_SIZE = 0.1
BIRD_OR_SPIDER = config["dataset"]["dataset_name"]
MODEL_NAME = config["orm"]["base_model_name"]
SEED = config["general"]["seed"]
RESUME_FROM_CHECKPOINT = config["orm"]["resume_from_checkpoint"]
ORM_OR_PRM = config["orm"]["orm_or_prm"]
VERSION = config["orm"]["version_model_finetuned"]
DATASET_VERSION = config["orm"]["dataset_train_version"]
base_model_short_name = MODEL_NAME.split("/")[-1]
NAME_MODEL_FINETUNED = f"{base_model_short_name}-{ORM_OR_PRM}-{BIRD_OR_SPIDER}-finetuned-{VERSION}"
FINETUNING_ITERATIONS = config["orm"]["finetuning_iterations"]

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    transformers_set_seed(seed)
set_seed(SEED)



class PatchedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        outputs = model(**inputs)
        loss = outputs.loss

        if num_items_in_batch is not None:
            loss = loss

        return (loss, outputs) if return_outputs else loss





if MACHINE == "hpc":
    DATASET_PATH = f"{conf.scratch_dir}/dataset_{ORM_OR_PRM}_{BIRD_OR_SPIDER}_train_{DATASET_VERSION}/train"
else:
    DATASET_PATH = f"{conf.project_dir}/../../dataset_{ORM_OR_PRM}_{BIRD_OR_SPIDER}_train_{DATASET_VERSION}/train"

def format_prompt(row):
    if ORM_OR_PRM == "orm":
        prompt = config["orm"]["orm_prompt"]
        prompt = prompt.replace("{QUESTION}", row['question'] + "\n" + row['schema'] + "\n" + row['question'])
        prompt = prompt.replace("{SQL}", row['sql'])
        prompt = prompt.replace("{DATA}", row['data'])
    else:
        prompt = config["orm"]["prm_prompt"]
        prompt = prompt.replace("{QUESTION}", row['question'] + "\n" + row['schema'] + "\n" + row['question'])
        prompt = prompt.replace("{STEP}", row['sql'])

    label = "Yes" if row['label'] == 1 else "No"
    prompt = prompt + " " + label
    return prompt



def tokenize_function(examples):
    tokenized = tokenizer(examples['text'], padding="max_length", truncation=True, max_length=512)
    tokenized['labels'] = tokenized['input_ids'].copy()
    return tokenized



def get_penultimate_checkpoint(checkpoint_dir):
    checkpoints = [d for d in os.listdir(checkpoint_dir) if d.startswith("checkpoint-") and os.path.isdir(os.path.join(checkpoint_dir, d))]
 
    checkpoint_steps = []
    for checkpoint in checkpoints:
        try:
            step = int(checkpoint.replace("checkpoint-", ""))
            checkpoint_steps.append((checkpoint, step))
        except ValueError:
            continue
    
    checkpoint_steps.sort(key=lambda x: x[1], reverse=True)
    
    penultimate_checkpoint = os.path.join(checkpoint_dir, checkpoint_steps[0][0])
    return penultimate_checkpoint



if torch.cuda.is_available():
    torch.cuda.empty_cache()

dataset = load_from_disk(DATASET_PATH)
dataset = dataset.map(lambda row: {'text': format_prompt(row)})
labels = dataset['label']
print("[Check]: Label distribution: ", Counter(labels))



if MACHINE == "hpc":
    NAME_TOKENIZER = MODEL_NAME.lower().replace('/', '_') + '_tokenizer'
    path_tokenizer = f"{conf.project_dir}/../../hf_cache/{NAME_TOKENIZER}"
    tokenizer = AutoTokenizer.from_pretrained(path_tokenizer)
else:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


if "llama" in NAME_TOKENIZER:
    print("[Info]: [LLama model detected, setting PAD token as EOS token")
    tokenizer.pad_token = tokenizer.eos_token  # Use EOS token as padding for LLama models


tokenized_dataset = dataset.map(tokenize_function, batched=True)
train_test_split = tokenized_dataset.train_test_split(test_size=VALIDATION_SIZE, seed=SEED)
train_dataset = train_test_split['train']
eval_dataset = train_test_split['test']


lora_config = LoraConfig(
    r=config["orm"]["r"],
    lora_alpha=config["orm"]["lora_alpha"],
    target_modules=config["orm"]["target_modules"],
    lora_dropout=config["orm"]["lora_dropout"],
    bias=config["orm"]["bias"]
)



for checkpoint_idx in range(FINETUNING_ITERATIONS):
    print(f"[Progress]: Training iteration {checkpoint_idx + 1}/{FINETUNING_ITERATIONS}")

    output_dir = f"{conf.project_dir}/{NAME_MODEL_FINETUNED}"
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    if MACHINE == "hpc":
        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME,torch_dtype=torch.float16,device_map="auto", local_files_only=True)
    else:
        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME,torch_dtype=torch.float16,device_map="auto")
    
    if "32B" in MODEL_NAME:
        model.gradient_checkpointing_enable()
    
    model = get_peft_model(model, lora_config)
    
    num_train_epochs = config["orm"]["num_train_epochs"]
    if config["general"]["debug"] == "true":
        num_train_epochs = 0.01

    training_args = TrainingArguments(
        report_to=[],
        fp16=config["orm"]["fp16"],
        output_dir=output_dir,
        eval_strategy="no",
        learning_rate=float(config["orm"]["learning_rate"]),
        per_device_train_batch_size=int(config["orm"]["per_device_train_batch_size"]),
        eval_accumulation_steps=1,
        gradient_accumulation_steps=1, 
        num_train_epochs=num_train_epochs,          
        weight_decay=0.01,
        save_strategy="epoch",
        save_total_limit=2,
        greater_is_better=False,
        logging_dir="./logs",
        logging_steps=10,
    )
    
    if "32B" in MODEL_NAME:
        trainer = PatchedTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
        )
    else:
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
        )
    
    if RESUME_FROM_CHECKPOINT:
        checkpoint_path = get_penultimate_checkpoint(output_dir)
        if checkpoint_path:
            print(f"[Progress]: Last checkpoint: {checkpoint_path}")
        trainer.train(resume_from_checkpoint=checkpoint_path)
    else:
        trainer.train()

    loss_list = [log['loss'] for log in trainer.state.log_history if 'loss' in log]

    loss_file = f"{output_dir}/training_loss_iter_{checkpoint_idx + 1}.json"
    with open(loss_file, "w") as f:
        json.dump({"iteration": checkpoint_idx + 1, "training_loss": loss_list}, f, indent=4)

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    del model
    del trainer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()