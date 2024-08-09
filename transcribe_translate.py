from whisper import load_model
from deep_translator import GoogleTranslator
import os
import json


def chunk_text(text, max_chars):
    chunks = []
    while len(text) > max_chars:
        # Find the last space within the max_chars limit to avoid cutting words
        split_point = text.rfind(' ', 0, max_chars)
        if split_point == -1:
            split_point = max_chars
        chunks.append(text[:split_point])
        text = text[split_point:].lstrip()
    if text:
        chunks.append(text)
    return chunks


def translate_text_chunks(chunks):
    translated_chunks = []
    for chunk in chunks:
        try:
            translated_text = GoogleTranslator(source='auto', target='en').translate(chunk)
            translated_chunks.append(translated_text)
        except Exception as e:
            print(f"Translation error: {e}")
            translated_chunks.append("")
    return ' '.join(translated_chunks)


model = load_model("base")
transcriptions = {}

audio_folder = 'audio_extracted'
max_chars = 4000

for file in os.listdir(audio_folder):
    if file.endswith('.mp3'):
        audio_path = os.path.join(audio_folder, file)
        result = model.transcribe(audio_path)
        text = result['text']

        chunks = chunk_text(text, max_chars)

        # Translate each chunk separately and combine the results
        translated_text = translate_text_chunks(chunks)

        transcriptions[file] = {'original': text, 'translated': translated_text}

# Save transcriptions to a JSON file
with open('transcriptions.json', 'w') as f:
    json.dump(transcriptions, f, indent=4)



