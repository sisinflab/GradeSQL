import os
import random
import numpy as np
import torch

from transformers import AutoTokenizer, TrainingArguments, Trainer, AutoModelForSequenceClassification, DataCollatorWithPadding
from transformers import set_seed as transformers_set_seed
from datasets import load_from_disk, DatasetDict
import evaluate

from config.config import Config



"""
Fine-tune a sequence classification model on a custom dataset using Hugging Face Transformers.

This script performs the following steps:
- Loads configuration parameters and dataset from disk.
- Formats dataset prompts based on ORM or PRM settings.
- Loads and prepares a pretrained tokenizer and model for sequence classification.
- Tokenizes the dataset and splits it into training and validation sets.
- Defines training arguments and metric computation (accuracy and ROC AUC).
- Uses Hugging Face Trainer API to fine-tune the model.
- Saves the fine-tuned model and tokenizer to the specified output directory.
"""

conf = Config()
config = conf.get_config()
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
MACHINE = conf.normal_or_hpc
VALIDATION_SIZE = 0.1
MODEL_NAME = config["orm"]["base_model_name"]
SEED = config["general"]["seed"]
RESUME_FROM_CHECKPOINT = config["orm"]["resume_from_checkpoint"]
ORM_OR_PRM = config["orm"]["orm_or_prm"]
VERSION = config["orm"]["version_model_finetuned"]
DATASET_VERSION = config["orm"]["dataset_train_version"]
base_model_short_name = MODEL_NAME.split("/")[-1]
BIRD_OR_SPIDER = config["dataset"]["dataset_name"]
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



if MACHINE == "hpc":
    DATASET_PATH = f"{conf.scratch_dir}/dataset_{ORM_OR_PRM}_{BIRD_OR_SPIDER}_train_{DATASET_VERSION}"
else:
    DATASET_PATH = f"{conf.project_dir}/../../dataset_{ORM_OR_PRM}_{BIRD_OR_SPIDER}_train_{DATASET_VERSION}"



def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True)



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

    return {"text": prompt, "label": row["label"]}



def compute_metrics(eval_pred):

    predictions, labels = eval_pred
    probabilities = np.exp(predictions) / np.exp(predictions).sum(-1, keepdims=True)

    # use probabilities of the positive class for ROC AUC
    positive_class_probs = probabilities[:, 1]
    auc = np.round(auc_score.compute(prediction_scores=positive_class_probs, references=labels)['roc_auc'],3)
    
    # predict most probable class
    predicted_classes = np.argmax(predictions, axis=1)
    acc = np.round(accuracy.compute(predictions=predicted_classes, references=labels)['accuracy'],3)
    
    return {"Accuracy": acc, "AUC": auc}



dataset_dict = load_from_disk(DATASET_PATH)
dataset_dict = dataset_dict.map(format_prompt)

NAME_TOKENIZER = MODEL_NAME.lower().replace('/', '_') + '_tokenizer'
path_tokenizer = f"{conf.project_dir}/../../hf_cache/{NAME_TOKENIZER}"
tokenizer = AutoTokenizer.from_pretrained(path_tokenizer)
tokenizer.pad_token = tokenizer.eos_token

id2label = {1: "Yes", 0: "No"}
label2id = {"Yes": 1, "No": 0}
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2, id2label=id2label, label2id=label2id,)
model.config.pad_token_id = tokenizer.pad_token_id

tokenized_data = dataset_dict.map(preprocess_function, batched=True)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

accuracy = evaluate.load("accuracy")
auc_score = evaluate.load("roc_auc")

num_train_epochs = config["orm"]["num_train_epochs"]
if config["general"]["debug"] == "true":
    num_train_epochs = 0.01

training_args = TrainingArguments(
    output_dir="output_dir",
    learning_rate=float(config["orm"]["learning_rate"]),
    per_device_train_batch_size=int(config["orm"]["per_device_train_batch_size"]),
    per_device_eval_batch_size=int(config["orm"]["per_device_train_batch_size"]),
    num_train_epochs=num_train_epochs,
    logging_strategy="epoch",
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
)



all_data = tokenized_data['train']
test_split = all_data.train_test_split(test_size=VALIDATION_SIZE, seed=SEED)
train_data = test_split['train']
validation_data = test_split['test']
tokenized_data = DatasetDict({
    'train': train_data,
    'validation': validation_data,
})



trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_data["train"],
    eval_dataset=tokenized_data["validation"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)
trainer.train()

output_dir = f"{conf.project_dir}/{NAME_MODEL_FINETUNED}"
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)