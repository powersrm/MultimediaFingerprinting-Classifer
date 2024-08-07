import hashlib
from sentence_transformers import SentenceTransformer
import json
import numpy as np
import re
from mutagen.mp3 import MP3

model = SentenceTransformer('all-MiniLM-L6-v2')


def preprocess_text(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'\W+', ' ', text)  # Remove punctuation and non-word characters
    return text


def generate_text_embedding(text):
    return model.encode(text)


def generate_fingerprint_from_embedding(embedding):
    return hashlib.sha256(embedding.tobytes()).hexdigest()


def get_audio_metadata(file_path):
    audio = MP3(file_path)
    duration = audio.info.length  # Duration in seconds
    return {
        'duration': duration
    }


# Load transcriptions from JSON file
try:
    with open('transcriptions.json', 'r') as f:
        transcriptions = json.load(f)
except FileNotFoundError:
    print("Transcriptions file not found.")
    transcriptions = {}
except json.JSONDecodeError:
    print("Error decoding JSON from transcriptions file.")
    transcriptions = {}

metadata = {}

for file, data in transcriptions.items():
    try:
        original_text = data['original']
        preprocessed_text = preprocess_text(original_text)
        embedding = generate_text_embedding(preprocessed_text)
        fingerprint = generate_fingerprint_from_embedding(embedding)

        # Path to the audio file (assumes it's in the same folder as the transcription file)
        audio_path = f'audio_extracted/{file}'
        audio_metadata = get_audio_metadata(audio_path)

        # Collect metadata
        metadata[file] = {
            'original_text_length': len(original_text),
            'embedding_dimensions': embedding.shape[0],
            'fingerprint': fingerprint,
            'duration': audio_metadata['duration']
        }
    except KeyError:
        print(f"Missing 'original' key in data for file: {file}")
    except Exception as e:
        print(f"An error occurred for file {file}: {e}")

# Save metadata to a JSON file
with open('metadata.json', 'w') as f:
    json.dump(metadata, f, indent=4)

print("Metadata extraction and fingerprint generation completed.")
