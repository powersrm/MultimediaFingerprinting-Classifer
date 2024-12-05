import os
import cv2
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from keras.applications.vgg16 import VGG16, preprocess_input
import json

def extract_frames(video_path, target_frames=300):
    """
    Extract frames from the video at specified intervals.
    """
    video = cv2.VideoCapture(video_path)
    fps = video.get(cv2.CAP_PROP_FPS)
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Calculate frame interval for the target number of frames
    frame_interval = total_frames / target_frames
    frames = []
    
    for i in range(target_frames):
        frame_position = int(i * frame_interval)
        video.set(cv2.CAP_PROP_POS_FRAMES, frame_position)
        ret, frame = video.read()
        if ret:
            frames.append(frame)
        else:
            print(f"Warning: Failed to read frame at position {frame_position}")
    
    video.release()
    
    if len(frames) == 0:
        print(f"Warning: No frames extracted from video: {video_path}")
    
    return frames, fps, total_frames

def extract_features(frames):
    """
    Extract features from the frames using the VGG16 model.
    """
    vgg16 = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    features = []

    for i, frame in enumerate(frames):
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (224, 224))
        img = np.expand_dims(img, axis=0)
        img = preprocess_input(img)
        
        feature = vgg16.predict(img)
        feature = feature.flatten()  # Flatten the feature map
        features.append(feature)
        
        if len(feature) == 0:
            print(f"Warning: No features extracted from frame {i}")
    
    if len(features) == 0:
        print("Warning: No features extracted from any frame.")
    
    return features

def compare_videos(video1_features, video2_features):
    """
    Compute cosine similarity between two sets of video features.
    Only compute similarity if features are non-empty.
    """
    if len(video1_features) == 0 or len(video2_features) == 0:
        print("Warning: One or both videos have empty feature sets.")
        return 0  # Return a default value (e.g., 0 for no similarity)
    
    # Flatten the feature arrays to 1D using the .flatten() method
    video1_features_flat = np.array(video1_features).flatten()
    video2_features_flat = np.array(video2_features).flatten()

    # Print feature shapes for debugging
    print(f"Shape of video1 features: {video1_features_flat.shape}")
    print(f"Shape of video2 features: {video2_features_flat.shape}")

    # Calculate cosine similarity
    similarity = cosine_similarity([video1_features_flat], [video2_features_flat])[0][0]
    print(f"Similarity calculated: {similarity}")
    return similarity

def process_all_videos(video_folder):
    """
    Process all videos in the folder sequentially (without batching).
    """
    # Get list of video files in the folder
    video_files = [f for f in os.listdir(video_folder) if f.endswith('.mp4')]
    total_videos = len(video_files)
    
    similarity_data = {}
    video_features = {}

    # Process each video in the folder
    for video_file in video_files:
        print(f"Processing video: {video_file}")
        video_path = os.path.join(video_folder, video_file)
        
        # Extract frames from the video
        frames, fps, total_frames = extract_frames(video_path)
        
        if len(frames) > 0:
            # Extract features for the frames
            features = extract_features(frames)
            video_features[video_file] = features
        else:
            print(f"Warning: No frames extracted from video {video_file}")
    
    # Compare each video with all others (including previously processed videos)
    for video1, features1 in video_features.items():
        for video2, features2 in video_features.items():
            if video1 != video2:
                similarity = compare_videos(features1, features2)
                # Store similarity data with actual video names
                similarity_data[f"{video1} vs {video2}"] = float(similarity)  # Convert similarity to native float

    # Save the results to a JSON file
    save_similarity_results(similarity_data)

def save_similarity_results(similarity_data):
    """
    Save the similarity results to a JSON file.
    """
    output_file = "similarity_data.json"
    with open(output_file, 'w') as f:
        json.dump(similarity_data, f, indent=4)
    print(f"Similarity data saved to {output_file}")

if __name__ == "__main__":
    video_folder = "D:/Projects/MultimediaFingerprinting-Classifer/audio_files"  # folder path
    process_all_videos(video_folder)  
