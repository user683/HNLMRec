from peft import PeftModel
from transformers import AutoModel, AutoTokenizer
import torch
import torch.nn.functional as F
from torch import Tensor
import json
import pickle
from tqdm import tqdm

# Average pooling function
def average_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

# Function to generate detailed instructions
def get_detailed_instruct(task_description: str, query: str) -> str:
    return f'Instruct: {task_description}\nQuery: {query}'

print("Starting to load data")
# Model path
origin_model = r'D:\LLM for Rec\Meta-Llama-3-8B-Instruct'

# Load the base model
model = AutoModel.from_pretrained(
    origin_model,
    device_map="auto",  # Automatically assign devices
    torch_dtype=torch.float16  # Half-precision floating point
)

print("LoRA weights loaded successfully")
# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    origin_model,
    trust_remote_code=True  # For community models on Hugging Face Hub
)

# If using LoRA or other fine-tuning
# model = PeftModel.from_pretrained(model, r'D:\LLM for Rec\Llama3_lora')
try:
    model = PeftModel.from_pretrained(model, r'D:\LLM for Rec\checkpoint-936')
except Exception as e:
    print("No fine-tuned weights detected, loading the original model")

# Task description
task = ("You are a helpful AI assistant. Based on the user's embedding, user profile, and the embeddings and item profiles of the items they have interacted with, \
        generate the user's hard negative embedding.")

# Count total lines in the JSON file
with open("Yelp_train_data.json", "r", encoding="utf-8") as f:
    total_lines = sum(1 for _ in f)

# Initialize dictionary to store hard negative embeddings
hard_negative_dict = {}

# Process each line in the JSON file
with open("Yelp_train_data.json", "r", encoding="utf-8") as f:
    for line in tqdm(f, desc="Processing lines in train_data.json", total=total_lines):
        try:
            data = json.loads(line.strip())
            query = data["query"]
            user_id = data["user_id"]
            item_id = data["item_id"]
        except json.JSONDecodeError as e:
            print(f"Error parsing line: {line}, Error: {e}")
            continue

        # Generate detailed instructions
        queries = [get_detailed_instruct(task, query)]
        # Tokenize input text
        tokenizer.pad_token = tokenizer.eos_token
        batch_dict = tokenizer(queries, max_length=512, padding=True, truncation=True, return_tensors='pt')

        # Ensure input tensors are on the correct device
        device = model.device if hasattr(model, "device") else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        batch_dict = {k: v.to(device) for k, v in batch_dict.items()}

        # Model inference
        outputs = model(**batch_dict)

        # Compute embeddings
        embeddings = average_pool(outputs.last_hidden_state, batch_dict['attention_mask'])

        # Initialize dictionary key if not present
        if user_id not in hard_negative_dict:
            hard_negative_dict[user_id] = {}
        # Save embeddings to dictionary
        hard_negative_dict[user_id][item_id] = embeddings.cpu().numpy().tolist()  # Save as nested list format

# Save dictionary to file
with open('hard_negative_dict_Yelp_Llama-3-8B_lora.pkl', 'wb') as f:
    pickle.dump(hard_negative_dict, f)