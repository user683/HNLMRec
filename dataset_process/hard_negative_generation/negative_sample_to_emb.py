import json
import time
from openai import OpenAI
from tqdm import tqdm
import argparse

# You can use other language model to replace ChatGPT-3.5

# Initialize OpenAI client
client = OpenAI(api_key="sk-c5I9a24a954cbd1b362704da80fc9b7725d905ca08fbWgaR", base_url="https://api.gptsapi.net/v1")

# Set up argument parser
parser = argparse.ArgumentParser(description='manual to this script')
parser.add_argument("--num", type=str, default="2")
args = parser.parse_args()

start_time = time.time()

# Function to get GPT embeddings
def get_gpt_emb(prompt):
    embedding = client.embeddings.create(
        model="text-embedding-ada-002",
        input=prompt,
        encoding_format="float"
    ).data[0].embedding
    return embedding

# Function to load JSON data
def load_json(file_path):
    """Load JSON file and return data and its length."""
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
        data_len = len(data)
    return data, data_len

# Define colors for console output
class Colors:
    GREEN = '\033[92m'
    END = '\033[0m'

# Print processing message
print(Colors.GREEN + "Encoding Semantic Representation" + Colors.END)
print("---------------------------------------------------\n")

# Initialize results list
results = []

# Load JSON data
data, data_length = load_json(f"./generated_negative_sample/generated_negative_profiles_{args.num}.json")

# Process each item in the data
for i in tqdm(range(data_length), desc="Processing"):
    try:
        # Get embeddings for the hard negative sample
        response_emb = get_gpt_emb(data[i]["hard_negative"])
        try:
            # Construct result dictionary
            result = {
                "user_id": data[i]["user_id"],
                "item_id": data[i]["item_id"],
                "item_title": data[i]["item_title"],
                "hard_negative": response_emb,
            }
            results.append(result)
        except Exception as e:
            # Fallback if an error occurs
            result = {
                "user_id": data[i]["user_id"],
                "item_id": data[i]["item_id"],
                "item_title": data[i]["item_title"],
                "hard_negative": None,
            }
            results.append(result)
    except:
        # Fallback if an error occurs during embedding generation
        result = {
            "user_id": data[i]["user_id"],
            "item_id": data[i]["item_id"],
            "item_title": None,
            "hard_negative": None,
        }
        results.append(result)

# Save results to a file
with open(f'./generated_negative_emb/negative_emb_{args.num}.json', 'w', encoding='utf-8') as outfile:
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