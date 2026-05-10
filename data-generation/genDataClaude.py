import anthropic
import json

client = anthropic.Anthropic(
    # defaults to os.environ.get("ANTHROPIC_API_KEY")
    api_key="my_api_key",
)

def generate_text(prompt, problem_id, language):
    try:
        response = client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=1024, # change if needed
            messages=[
                {"role": "system", "content": "You are a university student working on a programming assignment for a course. Your task is to provide the coding solution to a problem using the most appropriate programming language and practices for the given scenario. Your output should be only the code, without any explanations. Ensure the code is functional, correct, and follows standard conventions for the specified language. Do not include any introductory text or output besides the code itself."},
                {"role": "user", "content": prompt}
            ]
            
        )
        return response.content
    except Exception as e:
        print(f"Generation failed - Problem ID: {problem_id}, Language: {language}, Error: {str(e)}")
        return ""

inputfile = "input.jsonl"
output_jsonl = "output.jsonl"

dataset = []
count = 0
failed_generation = []
with open(inputfile, 'r', encoding='utf-8') as file:
    for line in file:
        problem_data = json.loads(line)
        id = problem_data['id']
        problem_description = problem_data['problem']
        languages = ["C", "C++", "Java", "Python"]
        for language in languages:
            prompt = f"""Programming language: {language}.

            Problem description:
            {problem_description}"""

            generated_text = generate_text(prompt, id, language)
            if not generated_text:
                failed_generation.append({
                    "problem id": id, 
                    "language": language,
                    "count": count
                })
                continue

            entry = {
                "solution_id": f"claude-{count}",
                "problem_id": id,
                "language": language,
                "solution": generated_text,
                "model": "claude-3-5-sonnet-20241022",
                "prompt": "You are a university student working on a programming assignment for a course. Your task is to provide the coding solution to a problem using the most appropriate programming language and practices for the given scenario. Your output should be only the code, without any explanations. Ensure the code is functional, correct for the specified language. Do not include any introductory text or output besides the code itself.",
            }
            json_object = json.dumps(entry, indent=4)

            with open(output_jsonl, 'w', encoding='utf-8') as output_file:
                output_file.write(json_object)
            count += 1
            dataset.append(entry)

# log faied generations
if failed_generations:
    print("Summary of failed generations:\n")
    for fail in failed_generations:
        print(f"Failed - Problem ID: {fail['problem_id']}, Language: {fail['language']}, Count: {fail['count']}")

# json to jsonl
with open(output_jsonl, 'w', encoding='utf-8') as jsonlfile:
    for data in dataset:
        jsonlfile.write(json.dumps(data) + "\n")
print(f"Dataset saved to {output_jsonl}")


