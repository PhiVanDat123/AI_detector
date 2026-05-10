import json
import glob

input_files = [f"{i}.json" for i in range(4000)]
output_file = "output.jsonl"

with open(output_file, "w") as jsonl_file:
    for file in input_files:
        with open(file, "r") as json_file:
            data = json.load(json_file)
            data["solution"] = data["solution"][data["solution"].find('\n')+1:-3]
            if isinstance(data, dict):
                jsonl_file.write(json.dumps(data) + "\n")
            else:
                raise ValueError(f"Expected a list in {file}, but got {type(data)}")

print(f"Combined JSONL file saved to {output_file}")
