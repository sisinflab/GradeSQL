import json
from rapidfuzz import fuzz
import re
import os
from collections import Counter



"""
This script adapts the OmniSQL training data (BIRD and Spider) into a format compatible 
with our pipeline.

It merges the original training JSON files with OmniSQL’s released training data by 
aligning questions and enriching entries with their corresponding `input_seq` and 
database schema extracted from our dataset.

Key steps:
- Loads our training JSON file and OmniSQL’s training JSON file.
- Extracts natural language questions and database schemas from `input_seq`.
- Matches questions from our data to questions in OmniSQL’s data (case-insensitive).
- For BIRD, prepends any available `evidence` to the question text before matching.
- Copies matched `input_seq` and schema into the OmniSQL entries.
- Identifies and reports duplicate questions in the OmniSQL training data.
- Filters out unmatched entries and writes a merged JSON output.
"""



COMBINATIONS = [
    ("bird", "train"),
    ("spider", "train"),
]
FUZZY_THRESHOLD = 50



for BIRD_OR_SPIDER, TRAIN_OR_DEV in COMBINATIONS:
    print(f"\nProcessing {BIRD_OR_SPIDER} - {TRAIN_OR_DEV}...")

    if BIRD_OR_SPIDER == "bird":
        FIRST_JSON_FILE = "../../data/train_bird.json"
        SECOND_JSON_FILE = "../../data/bird/train/train.json"
        OUTPUT_JSON_FILE = "../../data/bird/train/train_merged.json"
    else:
        FIRST_JSON_FILE = "../../data/train_spider.json"
        SECOND_JSON_FILE = "../../data/spider/train_spider.json"
        OUTPUT_JSON_FILE = "../../data/spider/train_merged.json"

    with open(FIRST_JSON_FILE, 'r', encoding='utf-8') as f:
        first_data = json.load(f)

    with open(SECOND_JSON_FILE, 'r', encoding='utf-8') as f:
        second_data = json.load(f)

    second_questions = [entry.get('question', '').lower() for entry in second_data if entry.get('question')]
    second_question_counts = Counter(second_questions)
    second_duplicates = {q: c for q, c in second_question_counts.items() if c > 1}
    total_duplicate_entries = sum(c - 1 for c in second_question_counts.values() if c > 1)
    print(f"Total duplicate entries: {total_duplicate_entries}")
    if second_duplicates:
        print("Duplicates in second JSON:", list(second_duplicates.items()))

    i = 0
    first_data_count = len(first_data)
    second_data_count = len(second_data)
    print(f"Number of entries in first JSON ({FIRST_JSON_FILE}): {first_data_count}")
    print(f"Number of entries in second JSON ({SECOND_JSON_FILE}): {second_data_count}")

    def extract_question(input_seq):
        pattern = r"Question:\n(.*?)\n\nInstructions:"
        match = re.search(pattern, input_seq, re.DOTALL)
        if match:
            return match.group(1).strip()
        return None

    def extract_schema(input_seq):
        pattern = r"Database Schema:\n(.*?)\nThis schema"
        match = re.search(pattern, input_seq, re.DOTALL)
        if match:
            return match.group(1).strip()
        return None

    match_count = 0

    for entry in first_data:
        input_seq = entry.get('input_seq')
        if not input_seq:
            print("No input_seq found")
            continue

        extracted_question = extract_question(input_seq)
        if not extracted_question:
            print("No extracted question")
            continue

        extracted_schema = extract_schema(input_seq)
        exact_match_found = False
        best_fuzzy_match = None
        best_fuzzy_score = FUZZY_THRESHOLD

        for second_entry in second_data:
            second_question = second_entry.get('question')
            if not second_question:
                print("No question found in second_data entry:", second_entry)
                continue

            if BIRD_OR_SPIDER == "bird":
                evidence = second_entry.get('evidence')
                if evidence != "":
                    second_question = evidence + "\n" + second_question

            if not second_question:
                continue

            if extracted_question.lower() == second_question.lower():
                second_entry['input_seq'] = input_seq
                second_entry['schema'] = extracted_schema
                exact_match_found = True
                i += 1
                break

    os.makedirs(os.path.dirname(OUTPUT_JSON_FILE), exist_ok=True)
    filtered_output_data = [entry for entry in second_data if 'input_seq' in entry]

    with open(OUTPUT_JSON_FILE, 'w', encoding='utf-8') as f:
        json.dump(filtered_output_data, f, indent=2)

    print(f"Output saved to {OUTPUT_JSON_FILE}")
    output_data_count = len(filtered_output_data)
    print(f"Number of entries in output JSON ({OUTPUT_JSON_FILE}): {output_data_count}")