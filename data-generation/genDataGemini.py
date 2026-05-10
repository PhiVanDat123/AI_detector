import json
import google.generativeai as genai

genai.configure(api_key="YOUR_API_KEY")
model = genai.GenerativeModel("gemini-1.5-flash")
system_prompt = """
You are a university student working on a programming assignment for a course.
You are given a programming problem in a specific programming language.
Your task is to write a program to solve the problem. 
You must provide a complete and functional program that solves the problem, without any explanation or introductory text before providing the program. 
The program should be well-structured, properly indented, and may include comments only if necessary. 
The program must be written in the specified programming language. 
Use unique and creative variable names, and avoid using common or generic names.
You need to test the program to ensure it works correctly. 
The program should not contain comments for the code lines; it should include only the code itself.
"""
def generate_text(prompt, problem_id, language):
    try:
        chat = model.start_chat(
            history=[
                {"role": "model", "parts": system_prompt},
            ]
        )
        response = chat.send_message(prompt)
        return response.text
    except Exception as e:
        print(f"Generation failed - Problem ID: {problem_id}, Language: {language}, Error: {str(e)}")
        return ""

# read the input file
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
                    "problem_id": id,
                    "language": language,
                    "count": count,
                })
                continue       
            entry = {
                "solution_id": f"gemini-{count}",
                "problem_id": id,
                "language": language,
                "solution": generated_text,
                "model": "gemini-1.5-pro",
                "prompt": "You are a university student working on a programming assignment for a course. Your task is to provide the coding solution to a problem using the most appropriate programming language and practices for the given scenario. Your output should be only the code, without any explanations. Ensure the code is functional, correct, and follows standard conventions for the specified language. Do not include any introductory text or output besides the code itself.",
            }
            json_object = json.dumps(entry, indent=4)
            # Save individual solution
            with open(f"data\generated\gemini\{count}.json", "w") as outfile:
                outfile.write(json_object)
                print(f"Generated record {count}, problem ID: {id}, language: {language}")
            
            dataset.append(entry)
            count += 1

# log faied generations
if failed_generations:
    print("Summary of failed generations:\n")
    for fail in failed_generations:
        print(f"Failed - Problem ID: {fail['problem_id']}, Language: {fail['language']}, Count: {fail['count']}")

# Save all solutions to JSONL file
with open(output_jsonl, "w", encoding='utf-8') as f:
    for entry in dataset:
        f.write(json.dumps(entry) + "\n")
print(f"Dataset saved to {output_jsonl}")
