from huggingface_hub import login, upload_folder, create_repo
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

from config.config import Config



def upload_download_model(upload_or_download: str, base_model_name: str, model_name_finetuned: str):

    """
    Upload or download a fine-tuned PEFT model to/from Hugging Face Hub.
    Modify the `upload_or_download` parameter to "upload" for uploading a model,
    or "download" for downloading a model.

    This script handles both uploading a locally fine-tuned PEFT model and downloading
    a fine-tuned model from the Hugging Face Hub. It supports creating a private repo
    (if uploading) and manages the associated tokenizer and model files.
    """

    login(token=config["orm"]["hf_token"])
    HF_USERNAME = config["orm"]["hf_username"]

    if upload_or_download == "upload":
        try:
            repo_url = create_repo(repo_id=f"{HF_USERNAME}/{model_name_finetuned}", private=True, repo_type="model")
        except:
            print(["[Warning]: repo already exists."])

        model_dir = f"{conf.project_dir}/{model_name_finetuned}"
        upload_folder(repo_id=f"{HF_USERNAME}/{model_name_finetuned}", folder_path=model_dir, repo_type="model")
        base_model = AutoModelForCausalLM.from_pretrained(base_model_name, torch_dtype="auto", device_map="auto")
        peft_model = PeftModel.from_pretrained(base_model, model_dir)
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        
        peft_model.push_to_hub(f"{HF_USERNAME}/{model_name_finetuned}")
        tokenizer.push_to_hub(f"{HF_USERNAME}/{model_name_finetuned}")
    else:
        base_model = AutoModelForCausalLM.from_pretrained(base_model_name, torch_dtype="auto",device_map="auto")
        model = PeftModel.from_pretrained(base_model, f"{HF_USERNAME}/{model_name_finetuned}")
        tokenizer = AutoTokenizer.from_pretrained(f"{HF_USERNAME}/{model_name_finetuned}")
        model.save_pretrained(f"{conf.project_dir}/{model_name_finetuned}")
        tokenizer.save_pretrained(f"{conf.project_dir}/{model_name_finetuned}")



conf = Config()
config = conf.get_config()
ORM_OR_PRM = config["orm"]["orm_or_prm"]
# Modify this variable to "upload" or "download" to choose the operation
UPLOAD_OR_DOWNLOAD = "upload"
BASE_MODEL_NAME = config["orm"]["base_model_name"]
VERSION = config["orm"]["version_model_finetuned"]
DATASET_NAME = config["dataset"]["dataset_name"]
base_model_short_name = BASE_MODEL_NAME.split("/")[-1]
MODEL_NAME_FINETUNED = f"{base_model_short_name}-{ORM_OR_PRM}-{DATASET_NAME}-finetuned-{VERSION}"



if __name__ == "__main__":
    upload_download_model(UPLOAD_OR_DOWNLOAD, BASE_MODEL_NAME, MODEL_NAME_FINETUNED)
