import requests
import json

# Set your OpenAI API key
API_KEY = ""
API_URL = "https://api.openai.com/v1/chat/completions"

# Read metadata from JSON file
try:
    with open('metadata.json') as f:
        metadata = json.load(f)
except FileNotFoundError:
    print("Metadata file not found.")
    metadata = {}
except json.JSONDecodeError:
    print("Error decoding JSON from metadata file.")
    metadata = {}

# Prepare the data to send to the API
data = {
    "metadata": metadata
}

# Format the request for GPT-3.5-turbo
payload = {
    "model": "gpt-3.5-turbo",
    "messages": [
        {
            "role": "system",
            "content": "You are an assistant that helps optimize multimedia storage based on metadata and identifies similar assets."
        },
        {
            "role": "user",
            "content": (
                f"Here is the metadata: {json.dumps(data)}. "
                "Please analyze it to identify which assets are very similar based on the metadata. "
                "Additionally, provide suggestions on how to reduce storage size for these similar assets and "
                "recommend any other metadata that might be useful for identifying similar assets."
            )
        }
    ],
    "max_tokens": 1500
}

# Set the headers for the request
headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

# Send the request to the OpenAI API
try:
    response = requests.post(API_URL, headers=headers, json=payload)
    response.raise_for_status()  # Raise an exception for HTTP errors
    suggestions = response.json()

    # Print in readable format
    print("Suggestions for reducing storage size and identifying similar assets:")
    if 'choices' in suggestions and len(suggestions['choices']) > 0:
        text = suggestions['choices'][0]['message']['content']
        print(text)
    else:
        print("No suggestions received.")

except requests.exceptions.RequestException as e:
    print(f"An error occurred: {e}")

