from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import login
import torch

from config.config import Config



"""
Downloads specified HuggingFace LLM models and saves their tokenizers locally for enviroments with no internet access.
Specify in model_names the models you want to download.
This script:
1. Loads project configuration from `Config`.
2. Logs in to HuggingFace Hub using the token stored in the config file.
3. Iterates over a list of model names:
    - Loads each model in half-precision (`float16`) with automatic device mapping.
    - Loads the corresponding tokenizer.
    - Saves the tokenizer to a local cache directory for later use by other scripts.
"""



conf = Config()
config = conf.get_config()

login(token=config["orm"]["hf_token"])

# Modify this list to include the models you want to download
model_names =  ["seeklhy/OmniSQL-32B"]
for MODEL_NAME in model_names:
    NAME_TOKENIZER = MODEL_NAME.lower().replace('/', '_') + '_tokenizer'
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME,torch_dtype=torch.float16,device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.save_pretrained(f"{conf.project_dir}/../../hf_cache/{NAME_TOKENIZER}")