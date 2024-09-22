from whisper import load_model
from pydub import AudioSegment
from pydub.utils import make_chunks
from deep_translator import GoogleTranslator
import os
import json

# Load Whisper model (you can also use the 'large' version for better results) MAYBE - https://huggingface.co/openai/whisper-large-v3#:~:text=The%20Whisper%20large%2Dv3%20model,epochs%20over%20this%20mixture%20dataset.
model = load_model("large")

# Function to split audio into chunks
def split_audio(audio_path, chunk_length_ms=30000):  # 30-second chunks
    audio = AudioSegment.from_file(audio_path)
    chunks = make_chunks(audio, chunk_length_ms)
    return chunks

# Function to transcribe and translate each audio chunk
def transcribe_and_translate(audio_chunk, chunk_index, audio_folder, file_name):
    # Save each chunk as a temporary file
    chunk_name = os.path.join(audio_folder, f"{file_name}_chunk{chunk_index}.mp3")
    audio_chunk.export(chunk_name, format="mp3")

    # Use Whisper's automatic language detection
    result = model.transcribe(chunk_name)
    original_text = result['text']

    # Translate the text to English
    translated_text = ""
    try:
        translated_text = GoogleTranslator(source='auto', target='en').translate(original_text)
    except Exception as e:
        print(f"Translation error on chunk {chunk_index}: {e}")

    return original_text, translated_text

# Folder containing audio files
audio_folder = 'audio_extracted'
transcriptions = {}

# Process each file in the audio folder
for file in os.listdir(audio_folder):
    if file.endswith('.mp3'):
        audio_path = os.path.join(audio_folder, file)

        # Split the audio into 30-second chunks
        audio_chunks = split_audio(audio_path)

        file_transcriptions = {'original': [], 'translated': []}

        # Process each chunk
        for i, audio_chunk in enumerate(audio_chunks):
            original_text, translated_text = transcribe_and_translate(audio_chunk, i, audio_folder, file)

            file_transcriptions['original'].append(original_text)
            file_transcriptions['translated'].append(translated_text)

        # Combine the chunks back into one full transcription
        transcriptions[file] = {
            'original': ' '.join([text for text in file_transcriptions['original'] if isinstance(text, str)]),
            'translated': ' '.join([text for text in file_transcriptions['translated'] if isinstance(text, str)])
        }

# Save transcriptions to a JSON file
with open('transcriptions.json', 'w') as f:
    json.dump(transcriptions, f, indent=4)
