import os
import ffmpeg

input_folder = 'audio_files'
output_folder = 'audio_extracted'

os.makedirs(output_folder, exist_ok=True)

for file in os.listdir(input_folder):
    if file.endswith('.mp4'):
        input_path = os.path.join(input_folder, file)
        output_path = os.path.join(output_folder, file.replace('.mp4', '.mp3'))
        ffmpeg.input(input_path).output(output_path).run()
