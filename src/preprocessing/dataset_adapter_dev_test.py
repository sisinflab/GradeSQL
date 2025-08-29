import json
from rapidfuzz import fuzz
import os
import re


"""
This script adapts the OmniSQL dataset (from their official GitHub release) into a format 
compatible with our pipeline.

It processes multiple benchmarks (BIRD and Spider) 
across dev/test splits by merging the OmniSQL JSON files with our existing dataset files.

Key steps:
- Loads and parses the OmniSQL JSON files alongside our dataset JSON files.
- Matches SQL queries between datasets (exact or fuzzy matching via RapidFuzz).
- Adds the corresponding "input_seq" and extracts the database schema.
- Handles dataset-specific differences (e.g., "SQL" vs "query" field naming).
- For Spider test, generates a gold SQL file with query–db_id pairs.
- Outputs merged JSON files ready for downstream input–output modeling.

Intended use:
Run the script directly — it loops over predefined dataset/split combinations, 
merges the OmniSQL data, and writes adapted JSON files (plus gold SQL for Spider test).
"""



COMBINATIONS = [
    ("bird", "dev"),
    ("spider", "dev"),
    ("spider", "test"),
]
FUZZY_THRESHOLD = 80



for BIRD_OR_SPIDER, TRAIN_OR_DEV in COMBINATIONS:
    print(f"\nProcessing {BIRD_OR_SPIDER} - {TRAIN_OR_DEV}...")

    if BIRD_OR_SPIDER == "bird":
        FIRST_JSON_FILE = "../../data/dev_bird.json"
        SECOND_JSON_FILE = "../../data/bird/dev_20240627/dev.json"
        OUTPUT_JSON_FILE = "../../data/bird/dev_20240627/dev_merged.json"
    elif BIRD_OR_SPIDER == "spider":
        if TRAIN_OR_DEV == "dev":
            FIRST_JSON_FILE = "../../data/dev_spider.json"
            SECOND_JSON_FILE = "../../data/spider/dev.json"
            OUTPUT_JSON_FILE = "../../data/spider/dev_merged.json"
        elif TRAIN_OR_DEV == "test":
            FIRST_JSON_FILE = "../../data/test_spider.json"
            SECOND_JSON_FILE = "../../data/spider/test.json"
            OUTPUT_JSON_FILE = "../../data/spider/test_merged.json"
            OUTPUT_GOLD = "../../data/spider/test_gold.sql"

    if BIRD_OR_SPIDER == "bird":
        field_sql_name = "SQL"
    else:
        field_sql_name = "query"



    def read_json_file(file_path):
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File {file_path} does not exist.")
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)


    def find_exact_match(sql, first_entries):
        for entry in first_entries:
            if entry.get('output_seq') == sql:
                return entry.get('input_seq')
        return None
    

    def extract_schema(input_seq):
        pattern = r"Database Schema:\n(.*?)\nThis schema"
        match = re.search(pattern, input_seq, re.DOTALL)
        if match:
            return match.group(1).strip()
        return None



    def find_fuzzy_match(sql, first_entries):
        best_score = 0
        best_input_seq = None
        for entry in first_entries:
            output_seq = entry.get('output_seq', '')
            score = fuzz.ratio(sql, output_seq)
            if score > best_score and score >= FUZZY_THRESHOLD:
                best_score = score
                best_input_seq = entry.get('input_seq')
        return best_input_seq



    def main():
        i = 0
        first_entries = read_json_file(FIRST_JSON_FILE)
        second_entries = read_json_file(SECOND_JSON_FILE)

        if not isinstance(first_entries, list):
            first_entries = [first_entries]
        if not isinstance(second_entries, list):
            second_entries = [second_entries]
        
        print(len(first_entries), "first entries")
        print(len(second_entries), "second entries")
        gold_lines = []
        output_entries = []
        for second_entry in second_entries:
            new_entry = second_entry.copy()
            sql = second_entry.get(field_sql_name, '')
            input_seq = find_exact_match(sql, first_entries)
            if input_seq is None:
                print(f"Fuzzy search triggered {i}")
                i = i + 1
                input_seq = find_fuzzy_match(sql, first_entries)
            if input_seq is not None:
                new_entry['input_seq'] = input_seq
                new_entry['schema'] = extract_schema(input_seq)
            output_entries.append(new_entry)

            # For spider test, collect query and db_id for gold file
            if BIRD_OR_SPIDER == "spider" and TRAIN_OR_DEV == "test":
                query = second_entry.get('query', '')
                db_id = second_entry.get('db_id', '')
                if query and db_id:
                    query = query.replace('\t', ' ')
                    gold_lines.append(f"{query}\t{db_id}")

        try:
            with open(OUTPUT_JSON_FILE, 'w', encoding='utf-8') as f:
                json.dump(output_entries, f, indent=4)
            print(f"Output JSON file created and saved as '{OUTPUT_JSON_FILE}'.")
        except Exception as e:
            print(f"Error saving output file: {e}")

        
        if BIRD_OR_SPIDER == "spider" and TRAIN_OR_DEV == "test":
            try:
                with open(OUTPUT_GOLD, 'w', encoding='utf-8') as f:
                    f.write('\n'.join(gold_lines))
                print(f"Gold SQL file created and saved as '{OUTPUT_GOLD}'.")
            except Exception as e:
                print(f"Error saving gold SQL file: {e}")



    if __name__ == "__main__":
        main()