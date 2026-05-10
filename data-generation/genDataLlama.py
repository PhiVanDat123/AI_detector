import os
from openai import OpenAI
import jsonlines

system_content = "You are a university student working on a programming assignment for a course. Your task is to provide the coding solution to a problem using the most appropriate programming language and practices for the given scenario. Your output should be only the code, without any explanations. Ensure the code is functional, correct for the specified language. Do not include any introductory text or output besides the code itself."
languages = ['C', 'C++', 'Python', 'Java']
model_ai = "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo"
input_file = "data/original/code_problems.jsonl"
output_file = "data/generated/new/uncleaned/solution_llama3.1_first500.jsonl"

client = OpenAI(
  #change token_api 
  api_key=os.environ.get("TOGETHER_API_KEY_2"),
  base_url="https://api.together.xyz/v1",
)

data = []
with jsonlines.open(input_file) as file:
    for line in file:
       data.append(line)

def get_user_content(language, problem):
   return  f"Solving this problem in {language}: {problem}"

count=0
stop_process = False
for item in data[:500]:
  process = []
  if stop_process:
    break
  problem = item['problem']
  id = item['id']
  for language in languages:
    user_content = get_user_content(language, problem)
    try:
      response = client.chat.completions.create(
      model=model_ai,
      messages=[
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content},
        ]
      )
      count+=1
      process.append({
                    "solution_id": f"llama3.1-8b-{count}",
                    "problem_id": id,
                    "language": language,
                    "solution": response.choices[0].message.content,
                    "model": "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
                    "prompt": "You are a university student working on a programming assignment for a course. Your task is to provide the coding solution to a problem using the most appropriate programming language and practices for the given scenario. Your output should be only the code, without any explanations. Ensure the code is functional, correct for the specified language. Do not include any introductory text or output besides the code itself.",
                })
    except Exception as e:
      print(e)
      print(f"Error in {id} for {language}")
      stop_process = True
      break
  if process:
    with jsonlines.open(output_file, mode="a") as  file:
      file.write_all(process)

