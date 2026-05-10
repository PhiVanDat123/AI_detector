import json
import logging
from libs.config import settings
from chatter import Chatter  

chatter = Chatter(
    system_message=(
        "You are a university student working on a programming assignment for a course. "
        "Your task is to provide the coding solution to a problem using the most appropriate "
        "programming language and practices for the given scenario. Your output should be only "
        "the code, without any explanations. Ensure the code is functional, correct, and follows "
        "standard conventions for the specified language. Do not include any introductory text or output besides the code itself."
    )
)

inputfile = r"input.jsonl"
output_jsonl = r"output.jsonl"

dataset = []
failed_generation = []
count = 0

with open(inputfile, 'r', encoding='utf-8') as file:
    for line in file:
        problem_data = json.loads(line.strip())
        problem_id = problem_data['id']
        problem_description = problem_data['problem']
        languages = ["C", "C++", "Java", "Python"]

        for language in languages:
            prompt = f"""Programming language: {language}.
            
            Problem description:
            {problem_description}"""

            generated_text = chatter.chat(prompt)

            processed_code = chatter.postprocess_code_completion(generated_text, lan=language.lower())
            
            if not processed_code.strip():
                logging.error(f"Generation failed - Problem ID: {problem_id}, Language: {language}")
                failed_generation.append({
                    "problem_id": problem_id,
                    "language": language,
                    "count": count,
                })
                continue

            entry = {
                "solution_id": f"gpt4o-mini-{count}",
                "problem_id": problem_id,
                "language": language,
                "solution": processed_code.strip(),
                "model": "gpt-4o-mini",
                "prompt": chatter.system_message,
            }

            output_file = f"{count}.json"
            with open(output_file, "w", encoding="utf-8") as outfile:
                json.dump(entry, outfile, indent=4)
                print(f"Generated record {count}, Problem ID: {problem_id}, Language: {language}")

            dataset.append(entry)
            count += 1

if failed_generation:
    logging.error("Summary of failed generations:")
    for fail in failed_generation:
        logging.error(f"Failed - Problem ID: {fail['problem_id']}, Language: {fail['language']}, Count: {fail['count']}")

# Lưu tất cả kết quả thành công vào tệp JSONL
with open(output_jsonl, 'w', encoding='utf-8') as jsonlfile:
    for data in dataset:
        jsonlfile.write(json.dumps(data) + "\n")

print(f"Dataset saved to {output_jsonl}")

