import streamlit as st
import os
import tempfile
import torch
import librosa
import numpy as np
import moviepy.editor as mp
import matplotlib.pyplot as plt
import shutil
import pandas as pd
from tensorflow.keras.models import load_model
from model_loader import process_video, process_audio
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer
import srt
from datetime import timedelta
import json
from safetensors.torch import load_file  # Correct way to load .safetensors

# Load models
video_model = load_model(r"models/video_model.h5")
audio_model = load_model(r"models/audio_model.keras")

# Define emotion mappings for all models
video_emotion_map = {
    0: 'Angry',
    1: 'Disgust',
    2: 'Fear',
    3: 'Happy',
    4: 'Sad', 
    5: 'Surprise',
    6: 'Neutral'
}

# For audio model using probability outputs
audio_emotion_map = {
    0: 'Neutral',
    1: 'Happy',
    2: 'Sad',
    3: 'Angry',
    4: 'Fear',
    5: 'Disgust',
    6: 'Surprise'
}

# Text emotion mapping - adjust this based on your DistilBERT model's outputs
# Changed to 6 classes to match the pretrained model
text_emotion_map = {
    0: 'Sad',
    1: 'Happy',
    2: 'Love',
    3: 'Anger',
    4: 'Fear',
    5: 'Surprise'
    # Removed Neutral since the model only has 6 classes
}

# Alternative audio mapping (if returned as string keys)
audio_emotion_map_alt = {
    '01': 'Neutral',
    '03': 'Happy',
    '04': 'Sad',
    '05': 'Angry',
    '06': 'Fear',
    '07': 'Disgust',
    '08': 'Surprise'
}

# Function to convert numerical predictions to emotion names
def convert_predictions_to_labels(predictions, emotion_map, is_audio=False):
    if predictions is None or not isinstance(predictions, np.ndarray) or predictions.size == 0:
        return None
    
    # If predictions are softmax probabilities for each class
    if len(predictions.shape) > 1 and predictions.shape[1] > 1:
        # Get the indices of the maximum values (most likely emotion class)
        emotion_indices = np.argmax(predictions, axis=1)
        # Map indices to emotion names
        emotion_labels = [emotion_map[idx] for idx in emotion_indices]
        return emotion_labels
    
    # If predictions are already class indices
    else:
        # For audio, check if these are string format indices
        if is_audio and isinstance(predictions[0], (str, np.str_)):
            return [audio_emotion_map_alt[str(idx)] for idx in predictions]
        return [emotion_map[int(idx)] for idx in predictions]

def extract_subtitles(video_path=None, subtitle_file=None):
    if subtitle_file:
        return subtitle_file

    return None

def load_text_emotion_model(model_path, config_path):
    try:
        # Load config (JSON file)
        with open(config_path, "r") as f:
            config = json.load(f)  # Load JSON config as a dictionary

        # Load model weights (for .safetensors file)
        state_dict = load_file(model_path)  # Correct method for .safetensors

        # Initialize the model with the correct number of labels from config
        num_labels = config.get("num_labels", 6)  # Default to 6 labels if missing
        model = DistilBertForSequenceClassification.from_pretrained(
            "distilbert-base-uncased",  # Use base model
            num_labels=num_labels
        )
        model.load_state_dict(state_dict)  # Load weights

        # Load the tokenizer
        tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

        return model, tokenizer

    except Exception as e:
        print(f"Error loading text model: {str(e)}")
        return None, None


# Function to analyze text emotion using DistilBERT
def analyze_text_emotion(text_segments, model, tokenizer):
    """
    Analyze emotion in text segments using DistilBERT
    
    Args:
        text_segments: List of text segments to analyze
        model: DistilBERT model
        tokenizer: DistilBERT tokenizer
        
    Returns:
        Numpy array of emotion probabilities for each segment
    """
    if model is None or tokenizer is None:
        return None
        
    model.eval()
    results = []
    
    for text in text_segments:
        if not text:  # Skip empty segments
            # Create a placeholder prediction with equal probabilities
            results.append(np.ones(len(text_emotion_map)) / len(text_emotion_map))
            continue
            
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        
        with torch.no_grad():
            outputs = model(**inputs)
            
        probabilities = torch.nn.functional.softmax(outputs.logits, dim=1).numpy()[0]
        results.append(probabilities)
    
    return np.array(results)

# Function to fuse only video and audio emotions (bimodal)
def fuse_emotions_bimodal(video_predictions, audio_predictions):
    """
    Fuses video and audio emotion predictions to produce a single multimodal prediction.
    
    Args:
        video_predictions: numpy array of shape (n_segments, 7) with video emotion probabilities
        audio_predictions: numpy array of shape (n_segments, 7) with audio emotion probabilities
        
    Returns:
        List of combined emotion labels, combined probabilities array
    """
    # Define the desired uniform emotion order
    all_emotions = ['Neutral', 'Happy', 'Sad', 'Angry', 'Fear', 'Disgust', 'Surprise']
    
    # Define emotion mappings from model indices to all_emotions indices
    video_mapping = {
        0: 3,  # Angry -> index 3 in all_emotions
        1: 5,  # Disgust -> index 5 in all_emotions
        2: 4,  # Fear -> index 4 in all_emotions
        3: 1,  # Happy -> index 1 in all_emotions
        4: 2,  # Sad -> index 2 in all_emotions
        5: 6,  # Surprise -> index 6 in all_emotions
        6: 0,  # Neutral -> index 0 in all_emotions
    }
    
    audio_mapping = {
        0: 0,  # Neutral -> index 0 in all_emotions
        1: 1,  # Happy -> index 1 in all_emotions
        2: 2,  # Sad -> index 2 in all_emotions
        3: 3,  # Angry -> index 3 in all_emotions
        4: 4,  # Fear -> index 4 in all_emotions
        5: 5,  # Disgust -> index 5 in all_emotions
        6: 6,  # Surprise -> index 6 in all_emotions
    }
    
    # Process each segment
    n_segments = max(
        len(video_predictions) if video_predictions is not None else 0,
        len(audio_predictions) if audio_predictions is not None else 0
    )
    
    aligned_video_predictions = []
    aligned_audio_predictions = []
    combined_probabilities = []
    combined_emotions = []
    highest_probabilities = []
    
    for i in range(n_segments):
        # Initialize arrays with zeros for the 7 emotion classes
        aligned_video = np.zeros(7)
        aligned_audio = np.zeros(7)
        
        # Map video probabilities to the aligned order
        if video_predictions is not None and i < len(video_predictions):
            for src_idx, dest_idx in video_mapping.items():
                aligned_video[dest_idx] = video_predictions[i][src_idx]
        
        # Map audio probabilities to the aligned order
        if audio_predictions is not None and i < len(audio_predictions):
            for src_idx, dest_idx in audio_mapping.items():
                aligned_audio[dest_idx] = audio_predictions[i][src_idx]
        
        # Combine probabilities with appropriate weights (59% video, 41% audio)
        combined_probs = (0.59 * aligned_video) + (0.41 * aligned_audio)
        
        # Store the aligned and combined probabilities
        aligned_video_predictions.append(aligned_video)
        aligned_audio_predictions.append(aligned_audio)
        combined_probabilities.append(combined_probs)
        
        # Determine the final emotion from the combined probabilities
        final_emotion_index = np.argmax(combined_probs)
        combined_emotions.append(all_emotions[final_emotion_index])
        highest_probabilities.append(combined_probs[final_emotion_index])
    
    return combined_emotions, combined_probabilities, highest_probabilities

# Function to display combined results table with highest probability
def display_combined_results(st, combined_emotions, highest_probabilities, time_points):
    """
    Displays the combined emotion results in Streamlit
    
    Args:
        st: Streamlit instance
        combined_emotions: List of emotion labels
        highest_probabilities: List of highest probability values
        time_points: List of time points in seconds
    """
    st.subheader("Combined Video-Audio Emotion Analysis")
    
    # Create a DataFrame for the final emotions with highest probability values
    final_df = pd.DataFrame({
        'Time (s)': time_points,
        'Final Emotion': combined_emotions,
        'Highest Probability': [f"{prob*100:.2f}%" for prob in highest_probabilities]
    })
    
    st.dataframe(final_df)

# Main Streamlit app
st.markdown("""
# Department of Information Technology, NITK
## DL(IT353) Course Project
*Done by:* Arnav Santosh (221AI010) and Shreesha M (221AI037)  
*Under the guidance of:* Jaidhar C D
""")
st.title("Multimodal Emotion Recognition App")

# Model paths for DistilBERT
text_model_path = "./models/textresults/emotion_classifier/model.safetensors"
text_config_path = "./models/textresults/emotion_classifier/config.json"

# File uploaders
uploaded_video = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov"], key="video_uploader")
uploaded_subtitle = st.file_uploader("Upload subtitles (optional)", type=["vtt", "srt"], key="subtitle_uploader")

if uploaded_video:
    # Save the uploaded file temporarily
    temp_video_path = os.path.join(tempfile.gettempdir(), uploaded_video.name)
    with open(temp_video_path, "wb") as f:
        f.write(uploaded_video.read())
    
    st.video(temp_video_path)

    # Make a copy of the video for audio extraction
    temp_audio_video_path = temp_video_path.replace(".mp4", "_audio_source.mp4")
    shutil.copy(temp_video_path, temp_audio_video_path)

    # Save uploaded subtitles if provided
    subtitle_path = None
    if uploaded_subtitle:
        subtitle_path = os.path.join(tempfile.gettempdir(), uploaded_subtitle.name)
        with open(subtitle_path, "wb") as f:
            f.write(uploaded_subtitle.read())

    # Load DistilBERT text emotion model if subtitles are provided
    text_model = None
    text_tokenizer = None
    if subtitle_path:
        with st.spinner("Loading DistilBERT model..."):
            try:
                text_model, text_tokenizer = load_text_emotion_model(text_model_path, text_config_path)
                if text_model:
                    st.success("Text emotion model loaded successfully!")
            except Exception as e:
                st.error(f"Error loading DistilBERT model: {e}")

    # Process video
    with st.spinner("Processing Video..."):
        video_predictions = process_video(temp_video_path, video_model)
        video_emotion_labels = convert_predictions_to_labels(video_predictions, video_emotion_map)

    # Run garbage collection to free memory before audio processing
    import gc
    gc.collect()

    # Extract and process audio
    with st.spinner("Extracting Audio..."):
        video_clip = mp.VideoFileClip(temp_audio_video_path)
        temp_audio_path = temp_video_path.replace(".mp4", ".wav")
        video_clip.audio.write_audiofile(temp_audio_path, verbose=False, logger=None)
        video_duration = video_clip.duration
        video_clip.close()  # Ensure file is closed

    with st.spinner("Processing Audio..."):
        audio_predictions = process_audio(temp_audio_path, audio_model)
        
        if audio_predictions is not None and isinstance(audio_predictions, np.ndarray) and audio_predictions.size > 0:
            audio_emotion_labels = convert_predictions_to_labels(audio_predictions, audio_emotion_map, is_audio=True)
        else:
            audio_emotion_labels = None

    # Process subtitles if available
    text_predictions = None
    text_emotion_labels = None

    if subtitle_path and text_model and text_tokenizer:
        with st.spinner("Processing Subtitles..."):
            try:
                # Read and parse SRT subtitles
                with open(subtitle_path, "r", encoding="utf-8") as f:
                    subtitles = list(srt.parse(f.read()))

                # Convert parsed subtitles into a list of text segments
                text_segments = [sub.content for sub in subtitles]  

                # Analyze text emotions
                text_predictions = analyze_text_emotion(text_segments, text_model, text_tokenizer)
                text_emotion_labels = convert_predictions_to_labels(text_predictions, text_emotion_map)

                st.success("Subtitle emotion analysis completed!")

            except Exception as e:
                st.error(f"Error processing subtitles: {e}")
                text_predictions = None
                text_emotion_labels = None
    # Show results
    st.write("## 📊 Results")

    # Three columns for Video, Audio, and Text Emotions
    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Video Emotions")
        if video_emotion_labels:
            video_df = pd.DataFrame({
                'Time (s)': [i*3 for i in range(len(video_emotion_labels))],
                'Emotion': video_emotion_labels
            })
            st.dataframe(video_df)
        else:
            st.warning("No video predictions available.")

    with col2:
        st.subheader("Audio Emotions")
        if audio_emotion_labels:
            audio_df = pd.DataFrame({
                'Time (s)': [i*3 for i in range(len(audio_emotion_labels))],
                'Emotion': audio_emotion_labels
            })
            st.dataframe(audio_df)
        else:
            st.warning("No audio predictions available.")

    with col3:
        st.subheader("Text Emotions")
        if text_emotion_labels:
            text_df = pd.DataFrame({
                'Text': text_segments,
                'Emotion': text_emotion_labels
            })
            st.dataframe(text_df)
        else:
            st.warning("No text predictions available or text model not loaded.")

    # Combined video-audio multimodal analysis (keeping text separate)
    if video_emotion_labels and audio_emotion_labels:
        time_points = [i*3 for i in range(max(
            len(video_predictions) if video_predictions is not None else 0,
            len(audio_predictions) if audio_predictions is not None else 0
        ))]
        
        # Fuse only video and audio emotions
        combined_emotions, combined_probabilities, highest_probabilities = fuse_emotions_bimodal(
            video_predictions, 
            audio_predictions
        )

        # Display the combined results table with highest probability
        display_combined_results(st, combined_emotions, highest_probabilities, time_points)

        # Multimodal Emotion Timeline
        st.subheader("Video-Audio Emotion Timeline")
        
        # Plot using matplotlib
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Define all emotions and assign unique numbers for plotting
        all_emotions = ['Neutral', 'Happy', 'Sad', 'Angry', 'Fear', 'Disgust', 'Surprise']
        emotion_to_num = {emotion: i for i, emotion in enumerate(all_emotions)}
        
        # Convert emotions to numbers for plotting
        emotion_nums = [emotion_to_num[e] for e in combined_emotions]
        
        # Plot line
        ax.plot(time_points, emotion_nums, 'o-', label='Combined Emotion', color='purple', linewidth=2)
        
        # Set y-ticks to emotion names
        ax.set_yticks(list(range(len(all_emotions))))
        ax.set_yticklabels(all_emotions)
        
        # Add labels and title
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Emotion')
        ax.set_title('Video-Audio Emotion Detection Timeline')
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Display the plot
        st.pyplot(fig)

        # Overall emotion analysis
        st.subheader("Video-Audio Emotion Analysis")
        
        emotion_counts = {}
        for emotion in combined_emotions:
            emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
        
        total_segments = len(combined_emotions)
        emotion_percentages = {emotion: (count/total_segments)*100 for emotion, count in emotion_counts.items()}
        
        dominant_df = pd.DataFrame({
            'Emotion': list(emotion_counts.keys()),
            'Count': list(emotion_counts.values()),
            'Percentage': [f"{emotion_percentages[emotion]:.1f}%" for emotion in emotion_counts.keys()]
        }).sort_values(by='Count', ascending=False)
        
        st.dataframe(dominant_df)
        
        # Pie chart of emotions
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.pie(
            list(emotion_counts.values()),
            labels=list(emotion_counts.keys()),
            autopct='%1.1f%%',
            startangle=90,
            colors=plt.cm.tab10.colors[:len(emotion_counts)]
        )
        ax.set_title('Distribution of Emotions Throughout Video')
        st.pyplot(fig)

        dominant_emotion = dominant_df.iloc[0]['Emotion']
        dominant_percentage = dominant_df.iloc[0]['Percentage']
        
        st.markdown(f"### Overall Dominant Emotion: {dominant_emotion} ({dominant_percentage})")

    # Separate text emotion analysis if available
    if text_emotion_labels:
        st.subheader("Text Emotion Analysis")
        
        # Count occurrences of each text emotion
        text_emotion_counts = {}
        for emotion in text_emotion_labels:
            text_emotion_counts[emotion] = text_emotion_counts.get(emotion, 0) + 1
        
        text_total_segments = len(text_emotion_labels)
        text_emotion_percentages = {emotion: (count/text_total_segments)*100 for emotion, count in text_emotion_counts.items()}
        
        text_dominant_df = pd.DataFrame({
            'Emotion': list(text_emotion_counts.keys()),
            'Count': list(text_emotion_counts.values()),
            'Percentage': [f"{text_emotion_percentages[emotion]:.1f}%" for emotion in text_emotion_counts.keys()]
        }).sort_values(by='Count', ascending=False)
        
        st.dataframe(text_dominant_df)
        
        # Pie chart of text emotions
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.pie(
            list(text_emotion_counts.values()),
            labels=list(text_emotion_counts.keys()),
            autopct='%1.1f%%',
            startangle=90,
            colors=plt.cm.tab10.colors[:len(text_emotion_counts)]
        )
        ax.set_title('Distribution of Text Emotions Throughout Video')
        st.pyplot(fig)
        
        text_dominant_emotion = text_dominant_df.iloc[0]['Emotion']
        text_dominant_percentage = text_dominant_df.iloc[0]['Percentage']
        
        st.markdown(f"### Dominant Text Emotion: {text_dominant_emotion} ({text_dominant_percentage})")
        
        # Text emotion timeline
        st.subheader("Text Emotion Timeline")
        time_points = [i*3 for i in range(len(text_emotion_labels))]
        
        # Define text emotions and assign unique numbers for plotting
        text_emotions = list(text_emotion_map.values())
        text_emotion_to_num = {emotion: i for i, emotion in enumerate(text_emotions)}
        
        # Convert emotions to numbers for plotting
        text_emotion_nums = [text_emotion_to_num[e] for e in text_emotion_labels]
        
        # Plot text emotion timeline
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(time_points, text_emotion_nums, '^-', label='Text Emotion', color='green', linewidth=2)
        
        # Set y-ticks to emotion names
        ax.set_yticks(list(range(len(text_emotions))))
        ax.set_yticklabels(text_emotions)
        
        # Add labels and title
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Emotion')
        ax.set_title('Text Emotion Detection Timeline')
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Display the plot
        st.pyplot(fig)

    # Comprehensive timeline with all modalities but keeping text separate
    if video_emotion_labels or audio_emotion_labels:
        st.subheader("Comprehensive Emotion Timeline")
        
        # Prepare data for plotting
        time_points = []
        video_emotions = []
        audio_emotions = []
        combined_emotions_list = []
        
        max_len = max(
            len(video_emotion_labels) if video_emotion_labels else 0,
            len(audio_emotion_labels) if audio_emotion_labels else 0
        )
        
        for i in range(max_len):
            time_sec = i * 3  # 3-second intervals
            time_points.append(time_sec)
            
            if video_emotion_labels and i < len(video_emotion_labels):
                video_emotions.append(video_emotion_labels[i])
            else:
                video_emotions.append(None)
                
            if audio_emotion_labels and i < len(audio_emotion_labels):
                audio_emotions.append(audio_emotion_labels[i])
            else:
                audio_emotions.append(None)
        
        # Get combined emotions if available
        if 'combined_emotions' in locals() and len(combined_emotions) > 0:
            combined_emotions_list = combined_emotions[:max_len]
            while len(combined_emotions_list) < max_len:
                combined_emotions_list.append(None)
        
        # Plot using matplotlib for better control over labels
        fig, ax = plt.subplots(figsize=(12, 7))
        
        # Define all emotions and assign unique numbers for plotting
        all_emotions = ['Neutral', 'Happy', 'Sad', 'Angry', 'Fear', 'Disgust', 'Surprise']
        emotion_to_num = {emotion: i for i, emotion in enumerate(all_emotions)}
        
        # Convert emotions to numbers for plotting
        video_nums = [emotion_to_num[e] if e is not None else np.nan for e in video_emotions]
        audio_nums = [emotion_to_num[e] if e is not None else np.nan for e in audio_emotions]
        
        # Plot lines
        ax.plot(time_points, video_nums, 'o-', label='Video Emotion', color='blue', alpha=0.7)
        ax.plot(time_points, audio_nums, 'x-', label='Audio Emotion', color='red', alpha=0.7)
        
        # Add combined emotions if available
        if 'combined_emotions' in locals() and len(combined_emotions) > 0:
            combined_nums = [emotion_to_num[e] if e is not None else np.nan for e in combined_emotions_list]
            ax.plot(time_points, combined_nums, 's-', label='Combined Emotion', color='purple', linewidth=2.5)
        
        # Set y-ticks to emotion names
        ax.set_yticks(list(range(len(all_emotions))))
        ax.set_yticklabels(all_emotions)
        
        # Add labels and title
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Emotion')
        ax.set_title('Comprehensive Video-Audio Emotion Detection Timeline')
        ax.legend(loc='upper right')
        ax.grid(True, linestyle='--', alpha=0.5)
        
        # Add a legend explaining the symbols
        from matplotlib.lines import Line2D
        custom_lines = [
            Line2D([0], [0], color='blue', marker='o', linestyle='-', markersize=8, label='Video'),
            Line2D([0], [0], color='red', marker='x', linestyle='-', markersize=8, label='Audio'),
        ]
            
        if 'combined_emotions' in locals() and len(combined_emotions) > 0:
            custom_lines.append(Line2D([0], [0], color='purple', marker='s', linestyle='-', markersize=8, linewidth=2.5, label='Combined'))
        
        second_legend = ax.legend(handles=custom_lines, loc='upper left', title="Emotion Sources")
        ax.add_artist(second_legend)
        
        # Add vertical grid lines for time segments
        ax.set_xticks(time_points)
        ax.set_xticklabels([f"{t}" for t in time_points], rotation=45)
        
        # Display the plot
        st.pyplot(fig)
        
        # Add explanation text
        st.markdown("""
        The combined line represents the final video-audio emotion based on weighted fusion:
        - 59% video weight
        - 41% audio weight
        
        Text emotions are analyzed separately due to different emotion categories.
        """)
    
    # Cleanup files safely
    try:
        os.remove(temp_video_path)
        os.remove(temp_audio_video_path)
        os.remove(temp_audio_path)
        if subtitle_path:
            os.remove(subtitle_path)
        st.success("Temporary files cleaned up successfully!")
    except Exception as e:
        st.error(f"Error during cleanup: {e}")
