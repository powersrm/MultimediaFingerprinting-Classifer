import requests
import json

# OpenAI API key
API_KEY = ""
API_URL = "https://api.openai.com/v1/chat/completions"

# Load metadata and similarities
try:
    with open('metadata.json', 'r') as f:
        metadata = json.load(f)
except FileNotFoundError:
    print("Metadata file not found.")
    metadata = {}
except json.JSONDecodeError:
    print("Error decoding JSON from metadata file.")
    metadata = {}

try:
    with open('similarities.json', 'r') as f:
        similarities = json.load(f)
except FileNotFoundError:
    print("Similarities file not found.")
    similarities = {}
except json.JSONDecodeError:
    print("Error decoding JSON from similarities file.")
    similarities = {}

def filter_similarities(similarities, threshold=0.9):
    """Filters similarities based on a threshold."""
    return {pair: score for pair, score in similarities.items() if score >= threshold}

# Filter similarities with threshold of 0.9
filtered_similarities = filter_similarities(similarities, threshold=0.9)

# Combine metadata and similarities into a single request, and truncate if necessary
metadata_str = json.dumps(metadata)
similarities_str = json.dumps(filtered_similarities)

# Truncate metadata and similarities if too large
max_message_tokens = 8000 - 1000  # Reserve tokens for the completion and system message

if len(metadata_str) + len(similarities_str) > max_message_tokens:
    # Simple truncation (could also use a more sophisticated method if needed)
    metadata_str = metadata_str[:max_message_tokens//2] + "..."
    similarities_str = similarities_str[:max_message_tokens//2] + "..."

combined_content = (
    f"Metadata: {metadata_str}, "
    f"Similarities: {similarities_str}. "
    "Analyze this data to identify all pairs of assets that have a similarity score of 0.9 or higher. "
    "List each pair explicitly and provide suggestions on how to reduce storage size for these similar assets. "
    "Also, recommend any additional metadata that might help in identifying similar assets."
)

# Set headers for the request
headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

# Prepare the payload for GPT-4
payload = {
    "model": "gpt-4",
    "messages": [
        {
            "role": "system",
            "content": "You are an assistant that helps optimize multimedia storage and identifies similar assets based on cosine similarity."
        },
        {
            "role": "user",
            "content": combined_content
        }
    ],
    "max_tokens": 2000
}

# Send the request to the OpenAI API
try:
    response = requests.post(API_URL, headers=headers, json=payload)
    response.raise_for_status()
    suggestions = response.json()

    # Print in readable format
    if 'choices' in suggestions and len(suggestions['choices']) > 0:
        text = suggestions['choices'][0]['message']['content']
        print(text)
    else:
        print("No suggestions received.")

except requests.exceptions.RequestException as e:
    if response.content:
        print("Response content:")
        print(response.content.decode())
