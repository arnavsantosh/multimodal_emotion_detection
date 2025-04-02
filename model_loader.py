import torch
import librosa
import numpy as np
import cv2
import os
import tensorflow as tf
from moviepy.editor import VideoFileClip
from tensorflow.keras.preprocessing.image import img_to_array
import subprocess

def preprocess_frame(frame):
    """Convert frame to grayscale, resize, normalize, and reshape for model."""
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)  #convert to grayscale
    frame = cv2.resize(frame, (48, 48))  #resize to (48x48)
    frame = frame.astype("float32") / 255.0  # normalize pixel values
    frame = img_to_array(frame)  
    frame = np.expand_dims(frame, axis=0)  #add batch dimension
    frame = np.expand_dims(frame, axis=-1)  # add channel dimension (1 for grayscale)
    return frame

def process_video(video_path, model):
    """Process video frames and get emotion predictions."""
    try:
        clip = VideoFileClip(video_path)
        frames = []
        duration = int(clip.duration)

        print(f"Video loaded. Duration: {duration} seconds")

        for t in range(0, duration, 3):  #sample at every 3-second interval
            try:
                frame = clip.get_frame(t + 1.5)  #extract midpoint frame
                preprocessed_frame = preprocess_frame(frame)
                frames.append(preprocessed_frame)
            except Exception as e:
                print(f"Error processing frame at {t+1.5}s: {e}")

        clip.close()  #ensure file is closed

        if not frames:
            print(" No frames extracted. Returning empty predictions.")
            return np.array([])

        frames = np.vstack(frames)  #stack frames into (N, 48, 48, 1)
        print(f" Frames stacked successfully. Shape: {frames.shape}")

        #run inference
        with tf.device("/GPU:0" if tf.config.list_physical_devices('GPU') else "/CPU:0"):
            print(" Running model prediction on video frames...")
            predictions = model.predict(frames)
        
        print(" Video prediction completed successfully.")
        return predictions

    except Exception as e:
        print(f" Error in process_video: {e}")
        return None

def extract_audio_segments(audio_path, sample_rate=22050, segment_duration=3):
    """Extract 3-second MFCC features from the audio file."""
    try:
        y, sr = librosa.load(audio_path, sr=sample_rate)
        total_duration = len(y) / sr
        num_segments = int(total_duration // segment_duration)

        features_list = []
        
        for i in range(num_segments):
            start_sample = i * segment_duration * sr
            end_sample = start_sample + (segment_duration * sr)
            segment = y[start_sample:end_sample]
            
            if len(segment) < segment_duration * sr:
                break  # Skip if segment is too short
            
            mfcc = librosa.feature.mfcc(y=segment, sr=sr, n_mfcc=18)
            mfcc_resized = np.zeros((18, 130))
            mfcc_resized[:, :min(130, mfcc.shape[1])] = mfcc[:, :130]
            
            mfcc_resized = np.expand_dims(mfcc_resized, axis=-1)
            features_list.append(mfcc_resized)
        
        if not features_list:
            print(" No valid audio segments extracted.")
            return None

        features_array = np.array(features_list)
        print("Extracted MFCC shape:", features_array.shape)
        return tf.convert_to_tensor(features_array, dtype=tf.float32)

    except Exception as e:
        print(f"Error extracting audio features from {audio_path}: {e}")
        return None

def process_audio(video_path, model):
    """Extracts audio using FFmpeg and predicts emotion for each 3s segment."""
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    audio_path = "temp_audio.wav"

    try:
        cmd = [
            "ffmpeg", "-i", video_path, "-vn",
            "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", audio_path, "-y"
        ]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print(" Audio successfully extracted using FFmpeg!")
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"FFmpeg audio extraction failed: {e}")

    features = extract_audio_segments(audio_path)
    os.remove(audio_path)  #cleanup temp file

    if features is not None:
        with tf.device("/GPU:0" if tf.config.list_physical_devices('GPU') else "/CPU:0"):
            print(" Running audio model prediction...")
            predictions = model.predict(features)
        
        print("Audio prediction completed successfully.")
        return predictions
    
    return None
