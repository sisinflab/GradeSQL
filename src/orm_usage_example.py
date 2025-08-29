import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel



prompt = """Question: What is the total horses record for each farm, sorted ascending?
CREATE TABLE competition_record (
    Competition_ID number, -- example: [1, 2]
    Farm_ID number, -- example: [2, 3]
    Rank number, -- example: [1, 2]
    PRIMARY KEY (Competition_ID),
    CONSTRAINT fk_competition_record_competition_id FOREIGN KEY (Competition_ID) REFERENCES farm_competition (Competition_ID),
    CONSTRAINT fk_competition_record_farm_id FOREIGN KEY (Farm_ID) REFERENCES farm (Farm_ID)
);

CREATE TABLE city (
    City_ID number, -- example: [1, 2]
    Status text, -- example: ['Town', 'Village']
    PRIMARY KEY (City_ID)
);

CREATE TABLE farm_competition (
    Competition_ID number, -- example: [1, 2]
    Host_city_ID number, -- example: [1, 2]
    PRIMARY KEY (Competition_ID),
    CONSTRAINT fk_farm_competition_host_city_id FOREIGN KEY (Host_city_ID) REFERENCES city (City_ID)
);

CREATE TABLE farm (
    Farm_ID number, -- example: [1, 2]
    Total_Horses number, -- example: [5056.5, 5486.9]
    Total_Cattle number, -- example: [8374.5, 8604.8]
    PRIMARY KEY (Farm_ID)
);
What is the total horses record for each farm, sorted ascending?
SQL: SELECT SUM(Total_Horses) AS Total_Horses, Farm_ID
FROM farm
GROUP BY Farm_ID
ORDER BY SUM(Total_Horses) ASC;
Is the SQL correct?"""

base_model = AutoModelForCausalLM.from_pretrained("seeklhy/OmniSQL-7B", torch_dtype="auto", device_map="auto")
peft_model = PeftModel.from_pretrained(base_model, "sisinflab-ai/GradeSQL-7B-ORM-Bird")
orm_model = peft_model.merge_and_unload()
orm_tokenizer = AutoTokenizer.from_pretrained("seeklhy/OmniSQL-7B", use_fast=True)

del base_model 
del peft_model  

inputs = orm_tokenizer(prompt, return_tensors="pt").to(orm_model.device)
        
with torch.no_grad():
    outputs = orm_model.generate(**inputs, max_new_tokens=1, return_dict_in_generate=True, output_scores=True, use_cache=False)

    generated_ids = outputs.sequences[0, len(inputs.input_ids[0]):]
    yes_token_id = orm_tokenizer.convert_tokens_to_ids("ĠYes")
    no_token_id = orm_tokenizer.convert_tokens_to_ids("ĠNo")
    
    yes_no_pos = None
    for i, token_id in enumerate(generated_ids):
        if token_id in [yes_token_id, no_token_id]:
            yes_no_pos = i
            break
    
    if yes_no_pos is None:
        print("[Warning]: No 'Yes' or 'No' token found in the generated output.")
        print("[Score]: 0.5")
    
    logits = outputs.scores[yes_no_pos]
    probs = torch.softmax(logits, dim=-1)
    yes_prob = probs[0, yes_token_id].item()
    generated_answer = "Yes" if generated_ids[yes_no_pos] == yes_token_id else "No"

    if generated_answer == "Yes":
        print("[Score]: ", yes_prob)
    elif generated_answer == "No":
        print("[Score]: ", 0)