import json
import os

from huggingface_hub import login
from datasets import Dataset, load_dataset

from config.config import Config
import glob



"""
Prepare and manage dataset for ORM finetuning, supporting both local files and Hugging Face Hub.

This script performs the following:
- Loads raw data from JSON files based on specified dataset (bird/spider), train/test split, and ORM/PRM mode.
- Collects positive and negative samples from corresponding directories, combining them with schema information.
- Formats data into a Hugging Face Dataset with 'question', 'schema', 'sql', 'data', 'label' (and 'step' for PRM).
- Computes and prints the distribution of labels in the dataset.
- Optionally uploads the processed dataset to the Hugging Face Hub or downloads an existing dataset from the Hub.
- Saves the processed dataset to disk for downstream use.
"""


conf = Config()
config = conf.get_config()
MACHINE = conf.normal_or_hpc # Machine used
TRAIN_OR_TEST = "train" # train or test
BIRD_OR_SPIDER = "spider"
PUSH_TO_HUB = False # If you want to upload to Hugging Face Hub
DOWNLOAD_FROM_HUB = True # If it is already present the dataset, download from the HUB
ORM_OR_PRM = "orm"
DATASET_VERSION = "v2"
HF_USERNAME = config["orm"]["hf_username"]

project_dir = conf.project_dir



if BIRD_OR_SPIDER == "bird":
    if TRAIN_OR_TEST == "train":
        dataset = f"{project_dir}/data/bird/train/train_merged.json"
    else:
        dataset = f"{project_dir}/data/bird/dev_20240627/dev_merged.json" 
elif BIRD_OR_SPIDER == "spider":
    if TRAIN_OR_TEST == "train":
        dataset = f"{project_dir}/data/spider/train_merged.json"
    else:
        dataset = f"{project_dir}/data/spider/dev_merged.json" 

negative_sample_path = f"{project_dir}/{ORM_OR_PRM}_negative_samples"
positive_samples_path = f"{project_dir}/{ORM_OR_PRM}_positive_samples"
output_path = f"{project_dir}/../../dataset_{ORM_OR_PRM}_{BIRD_OR_SPIDER}_{TRAIN_OR_TEST}_{DATASET_VERSION}"


if DOWNLOAD_FROM_HUB == False:

    data = []

    with open(dataset, "r") as f:
        data = json.load(f)

    formatted_data = []

    if ORM_OR_PRM == "orm":
        count = 0
        for id, sample in enumerate(data):
            json_id = id

            # Define the pattern for files matching {json_id}_*.json
            negative_pattern = os.path.join(negative_sample_path, f"{json_id}_*.json")
            positive_pattern = os.path.join(positive_samples_path, f"{json_id}_*.json")
            # Process negative samples
            for json_filepath in glob.glob(negative_pattern):
                if os.path.exists(json_filepath):
                    with open(json_filepath, "r") as f:
                        json_data = json.load(f)
                        for key, value in json_data.items():
                            formatted_data.append({
                                "question": value["question"],
                                "schema": sample["schema"],
                                "sql": value["sql"],
                                "data": value["data"],
                                "label": 0
                            })
                        

            # Process positive samples
            for json_filepath in glob.glob(positive_pattern):
                if os.path.exists(json_filepath):
                    with open(json_filepath, "r") as f:
                        json_data = json.load(f)
                        for key, value in json_data.items():
                            formatted_data.append({
                                "question": value["question"],
                                "schema": sample["schema"],
                                "sql": value["sql"],
                                "data": value["data"],
                                "label": 1
                            })
                        
    else:
        for id, sample in enumerate(data):
            
            json_id = id
            n_step = 0
            while True:
                json_filename = f"{json_id}_{n_step}.json"
                json_filepath = os.path.join(negative_sample_path, json_filename)
                print(json_filepath)
                if not os.path.exists(json_filepath):
                    break
                
                with open(json_filepath, "r") as f:
                    json_data = json.load(f)
                    for key, value in json_data.items():
                        if key == str(json_id):
                            formatted_data.append({
                                "question": value["question"],
                                "schema": sample["schema"],
                                "sql": value["sql"],
                                "step": value["step"],
                                "data": value["data"],
                                "label": 0
                            })
                n_step += 1
            
            n_step = 0
            while True:
                json_filename = f"{json_id}_{n_step}.json"
                json_filepath = os.path.join(positive_samples_path, json_filename)
                
                if not os.path.exists(json_filepath):
                    break 
                
                with open(json_filepath, "r") as f:
                    json_data = json.load(f)
                    for key, value in json_data.items():
                        if key == str(json_id):
                            formatted_data.append({
                                "question": value["question"],
                                "schema": sample["schema"],
                                "sql": value["sql"],
                                "step": value["step"],
                                "data": value["data"],
                                "label": 1
                            })
                n_step += 1


    dataset = Dataset.from_list(formatted_data)
    label_0_count = sum(1 for label in dataset['label'] if label == 0)
    label_1_count = sum(1 for label in dataset['label'] if label == 1)
    total_samples = len(dataset)
    label_0_percentage = (label_0_count / total_samples) * 100 if total_samples > 0 else 0
    label_1_percentage = (label_1_count / total_samples) * 100 if total_samples > 0 else 0

    print(f"Label 0 count: {label_0_count} ({label_0_percentage:.2f}%)")
    print(f"Label 1 count: {label_1_count} ({label_1_percentage:.2f}%)")
    
    if PUSH_TO_HUB == True:
        login(token=config["orm"]["hf_token"])  
        dataset.push_to_hub(f"{HF_USERNAME}/dataset_{ORM_OR_PRM}_{BIRD_OR_SPIDER}_{TRAIN_OR_TEST}_{DATASET_VERSION}", private=True)
else:
    dataset_id = f"{HF_USERNAME}/dataset_{ORM_OR_PRM}_{BIRD_OR_SPIDER}_{TRAIN_OR_TEST}_{DATASET_VERSION}"
    dataset = load_dataset(dataset_id)


dataset.save_to_disk(f"{output_path}")