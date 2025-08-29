import os
import json
import re
import torch
import random
import gc
from collections import Counter
from typing import List, Tuple
from transformers import AutoModelForCausalLM, pipeline
import torch.nn.functional as F

from utils.sql_utilities import *
from config.config import Config, setup_logging



def test_time_scaling(sql_predictions: List[str], db_id: str, question_evidence: str, strategy: str, gold: str = None, _id: str = None, orm_model: AutoModelForCausalLM = None, orm_tokenizer = None, question: str = None, schema: str = None) -> Tuple[str, int]:
    """
    Applies different SQL selection and evaluation strategies at test time.

    Depending on the specified strategy, this function:
    - Executes candidate SQL queries against the target database.
    - Optionally stores positive/negative samples for ORM fine-tuning.
    - Selects the best query based on execution correctness, majority voting,
      heuristic ranking, or ORM-based scoring.

    Args:
        sql_predictions (List[str]): List of SQL query candidates.
        db_id (str): Database identifier.
        question_evidence (str): Supporting information or evidence for the query.
        strategy (str): Selection strategy ("orm_finetuning_v1", "orm_finetuning_v2",
                        "none", "pass@k", "best_of_n_heuristic", "best_of_n_orm",
                        "majority_voting").
        gold (str, optional): Ground truth SQL query for comparison.
        _id (str, optional): Unique identifier for the current question/example.
        orm_model (AutoModelForCausalLM, optional): Model used for ORM-based ranking.
        orm_tokenizer (optional): Tokenizer for ORM model.
        question (str, optional): Original natural language question.
        schema (str, optional): Database schema description.

    Returns:
        Tuple[str, int] | str | List[str]: Depending on strategy:
            - A single SQL string (most cases)
            - A tuple with SQL and additional info
            - A list of SQL strings (for "pass@k")
    """

    logger = setup_logging()
    conf = Config()
    config = conf.get_config()
    
    TIMEOUT_SQL = config['inference']['timeout_sql']
    SEED = config['general']['seed']
    random.seed(SEED)

    match strategy.lower():

        case "orm_finetuning_v1":
            sql_results = []
            for sql in sql_predictions:
                result_sql = execute_query(sql, db_id, TIMEOUT_SQL)
                sql_results.append((sql, result_sql["score"], result_sql["data"]))

            gold_result = execute_query(gold, db_id, TIMEOUT_SQL)
            gold_result = (gold, gold_result["score"], gold_result["data"])

            if gold_result[1] == 0:
                print("[Warning]: Gold returns some errors. Skipping this question.")
                return ("Some SQL")


            negative_samples = []
            positive_samples = []

            for sql, score, data in sql_results:
                if sql != '':
                    if (score == 1 or score == 0.5) and data != gold_result[2]:
                        negative_samples.append((sql, score, data))
                    if score == 1 and data == gold_result[2]:
                        positive_samples.append((sql, score, data))

            if negative_samples and positive_samples:
                negative_samples.sort(key=lambda x: x[1], reverse=True)
                positive_samples.sort(key=lambda x: x[1], reverse=True)

                num_samples = min(len(negative_samples), len(positive_samples))

                os.makedirs(f"{conf.project_dir}/orm_negative_samples/", exist_ok=True)
                for i, (sql, score, data) in enumerate(negative_samples[:num_samples]):
                    checkpoint_results = {}
                    filename = f"{conf.project_dir}/orm_negative_samples/{_id}_{i}.json"
                    checkpoint_results[str(_id)] = {
                        "question": question_evidence,
                        "schema": schema,
                        "sql": sql,
                        "data": json.dumps(data)
                    }
                    with open(filename, "w") as f:
                        json.dump(checkpoint_results, f, indent=4)

                os.makedirs(f"{conf.project_dir}/orm_positive_samples/", exist_ok=True)
                for i, (sql, score, data) in enumerate(positive_samples[:num_samples]):
                    checkpoint_results = {}
                    filename = f"{conf.project_dir}/orm_positive_samples/{_id}_{i}.json"
                    checkpoint_results[str(_id)] = {
                        "question": question_evidence,
                        "schema": schema,
                        "sql": sql,
                        "data": json.dumps(data)
                    }
                    with open(filename, "w") as f:
                        json.dump(checkpoint_results, f, indent=4)

            return ("Some SQL")
        


        case "orm_finetuning_v2":

            
            sql_results = []
            for sql in sql_predictions:
                result_sql = execute_query(sql, db_id, TIMEOUT_SQL)
                sql_results.append((sql, result_sql["score"], result_sql["data"]))

            gold_result = execute_query(gold, db_id, TIMEOUT_SQL)
            gold_result = (gold, gold_result["score"], gold_result["data"])

            if gold_result[1] == 0:
                print("[Warning]: Gold returns some errors. Skipping this question.")
                return ("Some SQL")

            negative_samples = []
            positive_samples = []

            
            for sql, score, data in sql_results:
                if sql != '':
                    if (score == 1 or score == 0.5) and data != gold_result[2]:
                        negative_samples.append((sql, score, data))
                    if score == 1 and data == gold_result[2]:
                        positive_samples.append((sql, score, data))
            
            if negative_samples and positive_samples:
                negative_samples.sort(key=lambda x: x[1], reverse=True)
                positive_samples.sort(key=lambda x: x[1], reverse=True)

                os.makedirs(f"{conf.project_dir}/orm_negative_samples/", exist_ok=True)
                for i, (sql, score, data) in enumerate(negative_samples):
                    checkpoint_results = {}
                    filename = f"{conf.project_dir}/orm_negative_samples/{_id}_{i}.json"
                    checkpoint_results[str(_id)] = {
                        "question": question_evidence,
                        "schema": schema,
                        "sql": sql,
                        "data": json.dumps(data)
                    }
                    with open(filename, "w") as f:
                        json.dump(checkpoint_results, f, indent=4)

                os.makedirs(f"{conf.project_dir}/orm_positive_samples/", exist_ok=True)
                for i, (sql, score, data) in enumerate(positive_samples):
                    checkpoint_results = {}
                    filename = f"{conf.project_dir}/orm_positive_samples/{_id}_{i}.json"
                    checkpoint_results[str(_id)] = {
                        "question": question_evidence,
                        "schema": schema,
                        "sql": sql,
                        "data": json.dumps(data)
                    }
                    with open(filename, "w") as f:
                        json.dump(checkpoint_results, f, indent=4)

            return ("Some SQL")



        case "none":
            return sql_predictions[0]
        


        case "pass@k":
            return sql_predictions



        case "best_of_n_heuristic":

            scores = [execute_query(sql, db_id, TIMEOUT_SQL)["score"] for sql in sql_predictions]
            final_sql, max_score = max(zip(sql_predictions, scores), key=lambda x: x[1])
            return final_sql



        case "best_of_n_orm":
            final_sql = double_ranking_queries(sql_predictions, db_id, f"{question_evidence}\n{schema}\n{question_evidence}", question, orm_model, orm_tokenizer, _id)
            return final_sql
        


        case "majority_voting":
    
            successful_sqls = []
            for sql in sql_predictions:
                result = execute_query(sql, db_id, TIMEOUT_SQL)
                success = result["connection_successful"]
                rows = result["data"]
                if success:
                    hashable_rows = tuple(tuple(row) if isinstance(row, list) else row for row in rows)
                    successful_sqls.append((sql, hashable_rows))
            
            if not successful_sqls:
                return sql_predictions[0]
            
            rows_counts = Counter(pair[1] for pair in successful_sqls)

            if not rows_counts:
                return sql_predictions[0]
                
            most_common_rows = rows_counts.most_common(1)[0][0]
            for sql, row in successful_sqls:
                if row == most_common_rows:
                    return sql
            return successful_sqls[0][0]



def double_ranking_queries(queries: List[str], db_id: str, problem: str, question: str, orm_model: pipeline, orm_tokenizer, _id: str) -> str:

    """
    Selects the best SQL query using a combination of execution success and model-based ORM.

    Args:
        queries (List[str]): List of SQL queries.
        db_id (str): Database ID.
        problem (str): Schema + problem description.
        question (str): User natural language question.
        orm_model (pipeline): Fine-tuned LLM pipeline for judgment.

    Returns:
        str: Best query selected by double-ranking.
    """
        
    conf = Config()
    config = conf.get_config()
    timeout = config['inference']['timeout_sql']

    results = []
    for query in queries:
        result = execute_query(query, db_id, timeout)
        result = (query, result["score"], result["data"])
        results.append(result)

    queries, scores_heuristic, data = zip(*results)
    scores_similarities = score_orm(problem, question, queries, data, orm_model, orm_tokenizer, _id)
    all_candidates = list(zip(queries, scores_heuristic, scores_similarities))

    def select_best(candidates, target_score):
        filtered = [c for c in candidates if c[1] == target_score]
        if filtered:
            return max(filtered, key=lambda x: x[2])
        return None

    best = select_best(all_candidates, 1)
    if not best:
        best = select_best(all_candidates, 0.5)
    if not best:
        best = select_best(all_candidates, 0)

    best_query, best_heuristic, best_similarity = best

    return best_query



def score_orm(problem: str, question: str, queries: str, datas: str, orm_model: pipeline, orm_tokenizer, _id) -> List[int]:

    """
    Scores SQL queries based on LLM's judgment of semantic match.

    Args:
        problem (str): Prompt describing the problem.
        question (str): User question.
        queries (List[str]): SQL queries to score.
        orm_model (pipeline): HuggingFace pipeline used for inference.

    Returns:
        List[int]: List of scores ranging from 0 to 1.
    """

    torch.cuda.empty_cache()
    gc.collect()

    conf = Config()
    config = conf.get_config()
    MAX_NEW_TOKENS = config["orm"]["max_new_tokens"]
    FINETUNING_VERSION = config["orm"]["finetuning_version"]

    prompts = []
    for query, data in zip(queries, datas):
        prompt = config["orm"]["orm_prompt"]
        prompt = prompt.replace("{QUESTION}", problem)
        prompt = prompt.replace("{SQL}", query)
        prompt = prompt.replace("{DATA}", json.dumps(data))
        prompts.append(prompt)
    
    if config["orm"]["use_logits"] == True:
        scores = []
        for prompt, query, data in zip(prompts, queries, datas):
            if FINETUNING_VERSION == "v1":
                yes_prob = extract_yes_no_logits(prompt, orm_model, orm_tokenizer, MAX_NEW_TOKENS)
            elif FINETUNING_VERSION == "v2":
                yes_prob = extract_yes_no_sequence_classification(prompt, orm_model, orm_tokenizer, MAX_NEW_TOKENS)
            scores.append(yes_prob)

    return scores



def extract_yes_no(output: str) -> int:

    """Extracts a binary yes/no answer from a given string.

    Searches for the first occurrence of the word 'Yes' or 'No' (case-insensitive) in the input string.
    Returns 1 if 'Yes' is found, 0 if 'No' is found, or 0 if neither is found.

    Args:
        output (str): The input text from which to extract the answer.

    Returns:
        int: 1 if 'Yes' is detected, 0 otherwise.
    """

    match = re.search(r'\b(Yes|No)\b', output, re.IGNORECASE)
    if match:
        return 1 if match.group(1).lower() == 'yes' else 0
    
    print(f"[Warning]: no match found. Response malformed is {output}")
    return 1



def extract_yes_no_logits(prompt, model, tokenizer, max_new_tokens=10):
    
    """
    Estimates the probability that a model's generated response to a prompt is 'Yes'.
    Used when you have finetuned an ORM with version v1.

    This method generates a completition from the model and inspects the first
    occurrence of either the 'Yes' or 'No' token. It then returns the probability
    associated with 'Yes'.

    Args:
        prompt (str): Input text prompt.
        model: HuggingFace causal language model.
        tokenizer: Associated tokenizer.
        max_new_tokens (int): Maximum number of tokens to generate.

    Returns:
        float: Probability score between 0 and 1 for 'Yes',
               0 if 'No' is generated,
               0.5 if neither is found,
               or 0 if an error occurs.
    """

    try:
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        torch.cuda.empty_cache()
        gc.collect()
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, return_dict_in_generate=True, output_scores=True, use_cache=False)

        generated_ids = outputs.sequences[0, len(inputs.input_ids[0]):]
        yes_token_id = tokenizer.convert_tokens_to_ids("ĠYes")
        no_token_id = tokenizer.convert_tokens_to_ids("ĠNo")
        
        if yes_token_id == tokenizer.unk_token_id or no_token_id == tokenizer.unk_token_id:
            raise ValueError("[Error]: Tokenizer does not recognize 'Yes' or 'No' as single tokens.")
        
        yes_no_pos = None
        for i, token_id in enumerate(generated_ids):
            if token_id in [yes_token_id, no_token_id]:
                yes_no_pos = i
                break
        
        if yes_no_pos is None:
            print("[Warning]: No 'Yes' or 'No' token found in the generated output.")
            return 0.5
        
        logits = outputs.scores[yes_no_pos]
        probs = torch.softmax(logits, dim=-1)
        yes_prob = probs[0, yes_token_id].item()
        generated_answer = "Yes" if generated_ids[yes_no_pos] == yes_token_id else "No"

        if generated_answer == "Yes":
            confidence_score = yes_prob
            return confidence_score
        elif generated_answer == "No":
            return 0
    except Exception as e:
        print(f"[Error]: Failed to extract yes/no logits. Error: {e}")
        print(prompt)
        print(len(prompt))
        return 0
    



def extract_yes_no_sequence_classification(prompt, model, tokenizer, max_new_tokens=10):

    """
    Returns a probability score (between 0 and 1) indicating how likely the model thinks the answer is 'Yes'.
    Used when you have finetuned an ORM with version v2.

    Args:
        prompt: Tokenized prompt inputs.
        model: HuggingFace sequence classification model with {0: 'No', 1: 'Yes'} labels.
        tokenizer: Associated tokenizer.
        max_new_tokens (int): Not used here.

    Returns:
        float: Probability that the answer is 'Yes'.
    """

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = F.softmax(logits, dim=-1)

    yes_prob = probs[0, 1].item()
    return yes_prob