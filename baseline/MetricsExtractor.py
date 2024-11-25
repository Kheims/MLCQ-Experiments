import json
import os
import math

json_file = "MLCQCodeSmellSamples.json"
output_dir = "baseline/java_batches"
num_batches = 6

with open(json_file, "r") as f:
    data = json.load(f)

batch_size = math.ceil(len(data) / num_batches)

os.makedirs(output_dir, exist_ok=True)

for batch_index in range(num_batches):
    batch_folder = os.path.join(output_dir, f"batch_{batch_index + 1}")
    os.makedirs(batch_folder, exist_ok=True)

    start_index = batch_index * batch_size
    end_index = min((batch_index + 1) * batch_size, len(data))

    for idx, entry in enumerate(data[start_index:end_index], start=start_index):
        code_snippet = entry["code_snippet"]
        smell = entry["smell"]
        severity = entry["severity"]

        label = "_".join(smell.split()) if severity != "none" else "NoSmell"
        
        snippet_folder_name = f"{label}_{idx}"
        snippet_folder_path = os.path.join(batch_folder, snippet_folder_name)
        os.makedirs(snippet_folder_path, exist_ok=True)

        file_name = f"{snippet_folder_name}.java"
        file_path = os.path.join(snippet_folder_path, file_name)

        with open(file_path, "w") as java_file:
            java_file.write(code_snippet)

print(f"Created {num_batches} batches with Java files in '{output_dir}'")
