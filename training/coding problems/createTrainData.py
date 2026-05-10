import json
import random

file_paths = [
    "/Volumes/Workspace/Lab/AuthScan/data/original/filtered_solutions_16samples.jsonl",
    "/Volumes/Workspace/Lab/AuthScan/data/generated/new/cleaned/solution_gemini_1.5.jsonl",
    "/Volumes/Workspace/Lab/AuthScan/data/generated/new/cleaned/solution_gemini_2.0.jsonl",
    "/Volumes/Workspace/Lab/AuthScan/data/generated/new/cleaned/solution_gpt4o_mini.jsonl",
    "/Volumes/Workspace/Lab/AuthScan/data/generated/new/cleaned/solution_llama3.1_8b.jsonl",
    "/Volumes/Workspace/Lab/AuthScan/data/generated/new/cleaned/solution_gemini_1.5_first500.jsonl",
    "/Volumes/Workspace/Lab/AuthScan/data/generated/new/cleaned/solution_gemini_2.0_first500.jsonl",
    "/Volumes/Workspace/Lab/AuthScan/data/generated/new/cleaned/solution_gpt4o_mini_first500.jsonl",
    "/Volumes/Workspace/Lab/AuthScan/data/generated/new/cleaned/solution_llama3.1_8b_first500.jsonl"
]

labels = [0, 1, 1, 1, 1, 1, 1, 1, 1]
# Corresponding labels for each file (0 for human, 1 for machine-generated)
output_path = "/Volumes/Workspace/Lab/AuthScan/data/trainData/trainCombinedShuffled_500firstlast.jsonl"

def create_training_file(file_paths, labels, output):
    combined_data = []
    
    for file_path, label in zip(file_paths, labels):
        with open(file_path, 'r') as f:
            for line in f:
                data = json.loads(line)
                text = f"Language: {data['language']}\n\nCode:\n{data['solution']}"
                combined_data.append({"text": text, "label": label})
    
    random.shuffle(combined_data)
    
    with open(output, 'w') as out:
        for entry in combined_data:
            out.write(json.dumps(entry) + "\n")

create_training_file(file_paths, labels, output_path)
