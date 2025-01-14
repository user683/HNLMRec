import json
import numpy as np
import requests
import re
from tqdm import tqdm
import time


import argparse

parser = argparse.ArgumentParser(description='manual to this script')
parser.add_argument("--num", type=str, default="0")
args = parser.parse_args()

start_time = time.time()


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



system_prompt = ""
with open('user_system_prompt.txt', 'r') as f:
    for line in f.readlines():
        system_prompt += line


import json

example_prompts = {}
with open(f'./user_prompts_file/user_prompts_{args.num}.json', 'r', encoding='utf-8') as f:
    for line in f:
        try:
            # Parse each JSON line
            i_prompt = json.loads(line)
            # Store 'user_id' and cleaned 'prompt' in a dictionary
            user_id = i_prompt["user_id"]
            prompt = i_prompt['prompt'].replace(',{', '\n')
            # Add to dictionary
            example_prompts[user_id] = prompt
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON: {e}")


# print("------------------------------------------")
# print(example_prompts['A138IVRMXFBQCO'])


class Colors:
    GREEN = '\033[92m'
    END = '\033[0m'


results = []

print(Colors.GREEN + "Generating Profiles for Users" + Colors.END)
print("---------------------------------------------------\n")
print(Colors.GREEN + "The System Prompt (Instruction) is:\n" + Colors.END)
print(system_prompt)
print("---------------------------------------------------\n")


for user_id, prompt in tqdm(example_prompts.items(), desc="Processing prompts"):
    global content_data

    # print(Colors.GREEN + "The Input Prompt is:\n" + Colors.END)
    # print("==================================================\n")
    # print(user_id)
    # print(prompt)
    response = get_response_from_ollama(prompt)

    if response.status_code == 200:
        response_data = response.json()
        # print(response_data)
        try:
            content_json_str = response_data["message"]["content"]
            content_data = json.loads(content_json_str)
            # 假设响应中包含 summarization 和 reasoning 字段
            result = {
                "user_id": user_id,
                "summarization": content_data.get("summarization"),
                "reasoning": content_data.get("reasoning")
            }
            results.append(result)
            # print(Colors.GREEN + "Generated Results:\n" + Colors.END, result)
        except:
            result = {
                "user_id": user_id,
                "summarization": content_json_str,
                "reasoning": None
            }
            results.append(result)
            # print(Colors.GREEN + "Generated Results:\n" + Colors.END, result)
    else:
        print(f"Failed with status code {response.status_code}")


with open(f'./generated_user_profile/generated_user_profiles_{args.num}.json', 'w', encoding='utf-8') as outfile:
    json.dump(results, outfile, ensure_ascii=False, indent=4)
end_time = time.time()


elapsed_time = end_time - start_time


hours, rem = divmod(elapsed_time, 3600)
minutes, seconds = divmod(rem, 60)


formatted_time = f"{int(hours):02}:{int(minutes):02}:{int(seconds):02}"
print(f"数据处理完成，耗时：{formatted_time}")


