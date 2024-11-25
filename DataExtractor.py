import requests
import json
import time 
import os
from tqdm import tqdm


TOKEN = os.getenv('GITHUB_TOKEN')

def fetch_code_snippet(repo_url, commit_hash, file_path, start_line, end_line, request_count):

    if request_count > 4500:
        print("Reached 4500 requests, sleeping for 1 hour ...")
        time.sleep(3600)
        request_count = 0


    repo_name = repo_url.split('github.com:')[-1].replace('.git', '')
    
    raw_url = f"https://raw.githubusercontent.com/{repo_name}/{commit_hash}/{file_path.lstrip('/')}"

    headers = {'Authorization': f'token {TOKEN}'}
    
    response = requests.get(raw_url, headers=headers)
    
    if response.status_code == 200:
        lines = response.text.splitlines()
        
        code_snippet = "\n".join(lines[start_line-1:end_line])
        
        return code_snippet, request_count + 1
    else:
        print(f"Failed to fetch code from {raw_url} (status code: {response.status_code})")
        return None, request_count + 1 
    

def process_csv_and_save_to_json(csv_file, json_file,batch_size=50):
    
    request_count = 0
    json_data = []
    counter = 0

    with open(csv_file, 'r') as f:
        
        next(f)

        for line in tqdm(f, desc="Fetching code snippets"):
            parts = line.strip().split(";")
            _, _, _, smell, severity, _, type, code_name, repo_url, commit_hash, file_path, start_line, end_line, _, _ = parts
            
            start_line = int(start_line)
            end_line = int(end_line)
            
            code_snippet, request_count = fetch_code_snippet(repo_url, commit_hash, file_path, start_line, end_line, request_count)
            
            if code_snippet:
                json_data.append({
                    "repo_url": repo_url,
                    "commit_hash": commit_hash,
                    "file_path": file_path,
                    "start_line": start_line,
                    "end_line": end_line,
                    "code_snippet": code_snippet,
                    "smell": smell,
                    "severity": severity
                })

            counter += 1 
            if counter % batch_size == 0:
                save_json_data(json_file, json_data)
                json_data = []

        if json_data:
            save_json_data(json_file, json_data)
        print(f"Completed processing. Data saved to {json_file}")
    
    
def save_json_data(json_file, json_data):

    try:
        with open(json_file, 'r') as f:
            existing_data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        existing_data = []

    existing_data.extend(json_data)

    with open(json_file, 'w') as f:
        json.dump(existing_data, f, indent=4)

    print(f"Saved batch of {len(json_data)} entries to {json_file}")


csv_file = "MLCQCodeSmellSamples.csv"
json_file = "MLCQCodeSmellSamples"
process_csv_and_save_to_json(csv_file, json_file)
