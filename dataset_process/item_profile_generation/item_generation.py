import json
import numpy as np
import requests
import re
from tqdm import tqdm
import time

# Define URL and data
import argparse
parser = argparse.ArgumentParser(description='manual to this script')
parser.add_argument("--num", type=str, default="1")
args = parser.parse_args()

start_time = time.time()

# To achieve efficient parallel inference, our implementation is based
# on the Ollama platform. Readers may also opt for other parallel inference platforms, such as vLLM.

def get_response_from_ollama(prompt_normal):
    global system_prompt
    url = "http://localhost:11434/api/chat"
    payload = {
        "model": "llama3.1:8b",
        "messages": [
            {"role": "user", "content": system_prompt},
            {"role": "user", "content": prompt_normal}
        ],
        "stream": False
    }
    headers = {
        "Content-Type": "application/json"
    }

    response = requests.post(url, data=json.dumps(payload), headers=headers)

    return response

# Read the system prompt (instruction) to generate item profiles
system_prompt = ""
with open('item_system_prompt.txt', 'r') as f:
    for line in f.readlines():
        system_prompt += line

# Read example prompts for items
example_prompts = []
with open(f'./item_prompts_files/item_prompts_{args.num}.json', 'r', encoding='utf-8') as f:
    for line in f:
        try:
            # Parse each line as JSON
            i_prompt = json.loads(line)

            # Get and clean the 'prompt'
            prompt = i_prompt['prompt'].replace('""', '').replace(',', '').replace('    \n', '')
            example_prompts.append(prompt)
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON: {e}")

class Colors:
    GREEN = '\033[92m'
    END = '\033[0m'

results = []

print(Colors.GREEN + "Generating Profiles for Items" + Colors.END)
print("---------------------------------------------------\n")
print(Colors.GREEN + "The System Prompt (Instruction) is:\n" + Colors.END)
print(system_prompt)
print("---------------------------------------------------\n")

# Iteratively request each example_prompts
for i in tqdm(range(len(example_prompts)), desc="Processing prompts"):
    global content_data

    # Extract item_id from the prompt
    match = re.search(r'"item_id":\s*"(.+?)"', example_prompts[i])
    item_id = match.group(1)

    # Get response from the API
    response = get_response_from_ollama(example_prompts[i])

    if response.status_code == 200:
        response_data = response.json()
        try:
            content_json_str = response_data["message"]["content"]
            content_data = json.loads(content_json_str)
            # Assume the response contains summarization and reasoning fields
            result = {
                "item_id": item_id,
                "summarization": content_data.get("summarization"),
                "reasoning": content_data.get("reasoning")
            }
            results.append(result)
        except:
            result = {
                "item_id": item_id,
                "summarization": content_json_str,
                "reasoning": None
            }
            results.append(result)
    else:
        print(f"Failed with status code {response.status_code}")

# Save results to a file
with open(f'./item_generated_files/generated_item_profiles_{args.num}.json', 'w', encoding='utf-8') as outfile:
    json.dump(results, outfile, ensure_ascii=False, indent=4)

end_time = time.time()

# Calculate time difference
elapsed_time = end_time - start_time

# Convert to hours, minutes, and seconds
hours, rem = divmod(elapsed_time, 3600)
minutes, seconds = divmod(rem, 60)

# Format output as 00:00:00
formatted_time = f"{int(hours):02}:{int(minutes):02}:{int(seconds):02}"
print(f"Data processing completed. Time elapsed: {formatted_time}")