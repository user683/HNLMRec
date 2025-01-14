# Use dynamic parameter passing
import json
import numpy as np
import requests
import re
from tqdm import tqdm
import time
import argparse

parser = argparse.ArgumentParser(description='manual to this script')
parser.add_argument("--num", type=str, default="1")
args = parser.parse_args()

# Define URL and data
start_time = time.time()


def get_response_from_ollama(prompt_normal):
    global system_prompt
    url = "http://localhost:11434/api/chat"
    payload = {
        "model": "Qwen",
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
with open('negative_system_prompt.txt', 'r') as f:
    for line in f.readlines():
        system_prompt += line

# Read example prompts for items
example_prompts = {}
with open(f'./train_data/formatted_train_data_part_{args.num}.json', 'r', encoding='utf-8') as f:
    for line in f:
        inter_list = []
        try:
            # Parse each JSON line
            i_prompt = json.loads(line)

            # Store 'user_id' and cleaned 'prompt' in a dictionary
            inter_list.append(i_prompt["user_id"])
            inter_list.append(i_prompt["item_id"])
            inter_list.append(i_prompt["title"])
            prompt = i_prompt['prompt']
            # Add to dictionary
            example_prompts[tuple(inter_list)] = prompt
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON: {e}")


class Colors:
    GREEN = '\033[92m'
    END = '\033[0m'


results = []

print(Colors.GREEN + "Generating Profiles for Users" + Colors.END)
print("---------------------------------------------------\n")
print(Colors.GREEN + "The System Prompt (Instruction) is:\n" + Colors.END)
print(system_prompt)
print("---------------------------------------------------\n")

# Iteratively request each example_prompts
for inter_list, prompt in tqdm(example_prompts.items(), desc="Processing prompts"):
    # Make sure content_data is defined as None at the beginning of each loop
    content_data = None

    # Fetch response using the prompt
    response = get_response_from_ollama(prompt)

    if response.status_code == 200:
        response_data = response.json()

        try:
            # Extract content from response
            content_json_str = response_data["message"]["content"]
            content_data = json.loads(content_json_str)

            # Construct the result dictionary using extracted data
            result = {
                "user_id": inter_list[0],
                "item_id": inter_list[1],
                "item_title": inter_list[2],
                "hard_negative": content_data.get("hard_negative_item"),
                "reasoning": content_data.get("reasoning")
            }

        except (KeyError, json.JSONDecodeError) as e:
            # Fallback result if content extraction fails
            result = {
                "user_id": inter_list[0],
                "item_id": inter_list[1],
                "item_title": inter_list[2],
                "hard_negative": content_data.get("hard_negative_item") if content_data else None,
                "reasoning": content_data.get("reasoning") if content_data else None
            }

    else:
        # Print the error status if the request fails
        print(f"Failed with status code {response.status_code}")
        result = {
            "user_id": inter_list[0],
            "item_id": inter_list[1],
            "item_title": inter_list[2],
            "hard_negative": None,
            "reasoning": None
        }

    # Append the result to the results list
    results.append(result)

# Save results to a file
with open(f'./generated_negative_sample/generated_negative_profiles_{args.num}.json', 'w', encoding='utf-8') as outfile:
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