from paraphase import *
import jsonlines
import json
from openai import OpenAI
import random

model_list = {
    "deepseek-chat": "Deepseek V3", 
    "gpt-4o-mini": "GPT-4o mini", 
    "gemini-2.0-flash": "Gemini 2.0", 
    "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free": "Llama-3.3-70B-Instruct-Turbo"}
base_url_list = {
    "deepseek-chat": "https://api.deepseek.com", 
    "gpt-4o-mini": "https://api.openai.com/v1", 
    "gemini-2.0-flash": "https://generativelanguage.googleapis.com/v1beta/", 
    "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free": "https://api.together.xyz/v1"}

input_file = "input.jsonl"
success_output_file = "success.jsonl"
failed_output_file = "failed.jsonl"

data = []
with jsonlines.open(input_file) as file:
    for line in file:
        data.append(line)

success = []
failed = []
count=9244
for item in data[4622:5501]:
    title = item["file_name"]
    paragraph = item['content']
    original_id = item['ID']
    random_model = random.sample(list(model_list.keys()), 2)
    for model in random_model:
        try:
            new = paraphase_function(title, paragraph, original_id, model, base_url=base_url_list[model])
            entry = {
                "original_id": original_id,
                "new_id": count+1,
                "text": new,
                "label": "Human+AI",
                "label_detailed": f"human+{model_list[model]}"
            }
            json_object = json.dumps(entry, ensure_ascii=False, indent=4)
            with open(f"success\{count+1}.json", "w", encoding="utf8") as outfile:
                outfile.write(json_object)
                print(f"Generated record {count+1}, problem ID: {original_id}, model: {model}")
            success.append(entry)
            
        except Exception as e:
            item["model"] = model
            item["new_id"] = count+1
            json_obj = json.dumps(item, ensure_ascii=False, indent=4)
            failed.append(item)
            with open(f"failed\{count+1}.json", "w", encoding="utf8") as outfile:
                outfile.write(json_obj)
                print(f"Failed record {count+1}, problem ID: {original_id}, model: {model}")
        count+=1
