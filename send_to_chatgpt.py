import json
import requests

# OpenAI API key
API_KEY = ""
API_URL = "https://api.openai.com/v1/chat/completions"


def load_json(filename, fallback_value):
    """Loads JSON from a file, with fallback in case of errors."""
    try:
        with open(filename, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"{filename} file not found.")
        return fallback_value
    except json.JSONDecodeError:
        print(f"Error decoding JSON from {filename}.")
        return fallback_value


# Load data
metadata = load_json('metadata.json', {5})
video_similarities = load_json('video_similarities.json', {5})
scene_similarities = load_json('scene_similarities.json', {5})


def filter_similarities(similarities, threshold=0.99):
    """Filters similarities based on a threshold."""
    filtered = {pair: score for pair, score in similarities.items() if score >= threshold}
    print(f"Found {len(filtered)} pairs meeting the threshold of {threshold}.")
    return filtered


# Filter for identical video and scene pairs
filtered_video_similarities = filter_similarities(video_similarities, threshold=0.99)
filtered_scene_similarities = filter_similarities(scene_similarities, threshold=0.95)


def format_similarities(similarities, similarity_type):
    """Formats similarities into a human-readable string."""
    if similarities:
        return f"Essentially identical {similarity_type} pairs:\n" + "\n".join(
            [f"{pair}: {score}" for pair, score in similarities.items()])
    return f"No essentially identical {similarity_type} pairs found."


# Print filtered similarities
print(format_similarities(filtered_video_similarities, "video"))
print(format_similarities(filtered_scene_similarities, "scene"))

# Combine metadata, video similarities, and scene similarities into a request
metadata_str = json.dumps(metadata)
video_similarities_str = json.dumps(filtered_video_similarities)
scene_similarities_str = json.dumps(filtered_scene_similarities)

# Check total length and truncate if too large
max_message_tokens = 8000 - 1000  # Reserve tokens for the system message and completion
total_length = len(metadata_str) + len(video_similarities_str) + len(scene_similarities_str)

if total_length > max_message_tokens:
    # Truncate based on proportional length
    metadata_str = metadata_str[:int(max_message_tokens * (len(metadata_str) / total_length))] + "..."
    video_similarities_str = video_similarities_str[
                             :int(max_message_tokens * (len(video_similarities_str) / total_length))] + "..."
    scene_similarities_str = scene_similarities_str[
                             :int(max_message_tokens * (len(scene_similarities_str) / total_length))] + "..."

# Combine content for API request
combined_content = (
    f"Metadata: {metadata_str}, "
    f"Video Similarities: {video_similarities_str}, "
    f"Scene Similarities: {scene_similarities_str}. "
    "Analyze this data to identify all pairs of assets (both video and scene) with a similarity score of 0.95 or higher. "
    "List each pair explicitly and provide suggestions on how to reduce storage size for these similar assets. "
    "Also, recommend additional metadata that might help in identifying similar assets, focusing on attributes that can enhance retrieval efficiency."
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
    "max_tokens": 5000
}

# Send the request to the OpenAI API
try:
    response = requests.post(API_URL, headers=headers, json=payload)
    response.raise_for_status()
    suggestions = response.json()

    # Check and print suggestions
    if 'choices' in suggestions and len(suggestions['choices']) > 0:
        text = suggestions['choices'][0]['message']['content']
        print(text)
        # Optionally save suggestions to a new file
        with open('analysis_suggestions.json', 'w') as f:
            json.dump(text, f, indent=4)
    else:
        print("No suggestions received.")

except requests.exceptions.RequestException as e:
    print(f"Request failed: {e}")
    if response is not None and response.content:
        print("Response content:")
        print(response.content.decode())
