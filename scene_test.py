import os
import json
from scene_signature_generation import (
    build_autoencoder, prepare_training_data, process_videos, compare_signatures)

# Paths for testing
input_folder = 'audio_files'
output_folder = 'video_frames'
os.makedirs(output_folder, exist_ok=True)

def run_tests():
    input_shape = 64 * 64
    autoencoder, encoder = build_autoencoder(input_shape)
    training_data = prepare_training_data(input_folder)
    autoencoder.fit(training_data, training_data, epochs=5, batch_size=16)
    
    video_signatures = process_videos(input_folder, output_folder, (224, 224))  # Use 224x224 for VGG16
    frame_similarities = compare_signatures(video_signatures)
    
    with open('frame_similarities.json', 'w') as f:
        json.dump(frame_similarities, f, indent=4)
    
    print("Test frame-by-frame signature comparison and similarity generation completed.")

if __name__ == "__main__":
    run_tests()
