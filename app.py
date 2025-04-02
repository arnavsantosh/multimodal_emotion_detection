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
from safetensors.torch import load_file  #for loading .safetensors model weights

video_model = load_model(r"models\video_model.h5")
audio_model = load_model(r"models\audio_model.keras")

#map of numerical predictions to emotion labels
video_emotion_map = {
    0: 'Angry',
    1: 'Disgust',
    2: 'Fear',
    3: 'Happy',
    4: 'Sad', 
    5: 'Surprise',
    6: 'Neutral'
}

audio_emotion_map = {
    0: 'Neutral',
    1: 'Happy',
    2: 'Sad',
    3: 'Angry',
    4: 'Fear',
    5: 'Disgust',
    6: 'Surprise'
}

text_emotion_map = {
    0: 'Sad',
    1: 'Happy',
    2: 'Love',
    3: 'Anger',
    4: 'Fear',
    5: 'Surprise'
}

#alternative mapping for audio model predictions if returned as string keys
audio_emotion_map_alt = {
    '01': 'Neutral',
    '03': 'Happy',
    '04': 'Sad',
    '05': 'Angry',
    '06': 'Fear',
    '07': 'Disgust',
    '08': 'Surprise'
}

def convert_predictions_to_labels(predictions, emotion_map, is_audio=False):
    #makes sure input is valid before proceeding
    if predictions is None or not isinstance(predictions, np.ndarray) or predictions.size == 0:
        return None
    
    #handle softmax probability outputs 
    if len(predictions.shape) > 1 and predictions.shape[1] > 1:
        emotion_indices = np.argmax(predictions, axis=1)
        return [emotion_map[idx] for idx in emotion_indices]
    
    #handle cases where predictions are already class indices
    else:
        if is_audio and isinstance(predictions[0], (str, np.str_)):
            return [audio_emotion_map_alt[str(idx)] for idx in predictions]
        return [emotion_map[int(idx)] for idx in predictions]

def extract_subtitles(video_path=None, subtitle_file=None):
    #check if a separate subtitle file is provided
    if subtitle_file:
        return subtitle_file
    
    return None

def load_text_emotion_model(model_path, config_path):
    try:
        #load model config
        with open(config_path, "r") as f:
            config = json.load(f)

        #load model weights
        state_dict = load_file(model_path)

        #get number of labels from config, default to 6 if missing
        num_labels = config.get("num_labels", 6)
        model = DistilBertForSequenceClassification.from_pretrained(
            "distilbert-base-uncased", num_labels=num_labels
        )
        model.load_state_dict(state_dict)

        #load tokenizer
        tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

        return model, tokenizer
    except Exception as e:
        print(f"Error loading text model: {str(e)}")
        return None, None

def analyze_text_emotion(text_segments, model, tokenizer):
    #analyze emotion from text segments using distilbert
    if model is None or tokenizer is None:
        return None
        
    model.eval()
    results = []
    
    for text in text_segments:
        if not text:
            #if segment is empty, return equal probabilities for all emotions
            results.append(np.ones(len(text_emotion_map)) / len(text_emotion_map))
            continue
            
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        probabilities = torch.nn.functional.softmax(outputs.logits, dim=1).numpy()[0]
        results.append(probabilities)
    
    return np.array(results)

def fuse_emotions_bimodal(video_predictions, audio_predictions):
    #combines video and audio emotion predictions into a single output
    all_emotions = ['Neutral', 'Happy', 'Sad', 'Angry', 'Fear', 'Disgust', 'Surprise']
    
    #map indices from model-specific outputs to a common emotion order
    video_mapping = {0: 3, 1: 5, 2: 4, 3: 1, 4: 2, 5: 6, 6: 0}
    audio_mapping = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6}
    
    #determine the number of segments to process
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
        aligned_video = np.zeros(7)
        aligned_audio = np.zeros(7)
        
        if video_predictions is not None and i < len(video_predictions):
            for src_idx, dest_idx in video_mapping.items():
                aligned_video[dest_idx] = video_predictions[i][src_idx]
        
        if audio_predictions is not None and i < len(audio_predictions):
            for src_idx, dest_idx in audio_mapping.items():
                aligned_audio[dest_idx] = audio_predictions[i][src_idx]
        
        #weight video more heavily (59%) than audio (41%) according to base paper
        combined_probs = (0.59 * aligned_video) + (0.41 * aligned_audio)
        
        aligned_video_predictions.append(aligned_video)
        aligned_audio_predictions.append(aligned_audio)
        combined_probabilities.append(combined_probs)
        
        final_emotion_index = np.argmax(combined_probs)
        combined_emotions.append(all_emotions[final_emotion_index])
        highest_probabilities.append(combined_probs[final_emotion_index])
    
    return combined_emotions, combined_probabilities, highest_probabilities

def display_combined_results(st, combined_emotions, highest_probabilities, time_points):
    #show the final combined emotion predictions in a table
    st.subheader("Combined Video-Audio Emotion Analysis")
    final_df = pd.DataFrame({
        'Time (s)': time_points,
        'Final Emotion': combined_emotions,
        'Highest Probability': [f"{prob*100:.2f}%" for prob in highest_probabilities]
    })
    st.dataframe(final_df)

#main streamlit code
#title and project details
st.markdown("""
# Department of Information Technology
# National Institute of Technology Karnataka, Surathkal
## Deep Learning (IT353) Course Project
**Done by:** Arnav Santosh (221AI010) and Shreesha M (221AI037)  
**Under the guidance of:** Jaidhar C D
""")
st.title("Multimodal Emotion Recognition App")

#define model paths for emotion recognition
text_model_path = "./models/textresults/emotion_classifier/model.safetensors"
text_config_path = "./models/textresults/emotion_classifier/config.json"

#file uploaders for video and subtitles
uploaded_video = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov"], key="video_uploader")
uploaded_subtitle = st.file_uploader("Upload subtitles (optional)", type=["vtt", "srt"], key="subtitle_uploader")

if uploaded_video:
    #save video temporarily
    temp_video_path = os.path.join(tempfile.gettempdir(), uploaded_video.name)
    with open(temp_video_path, "wb") as f:
        f.write(uploaded_video.read())
    
    st.video(temp_video_path)

    #create a copy for audio extraction
    temp_audio_video_path = temp_video_path.replace(".mp4", "_audio_source.mp4")
    shutil.copy(temp_video_path, temp_audio_video_path)

    #save subtitles if uploaded
    subtitle_path = None
    if uploaded_subtitle:
        subtitle_path = os.path.join(tempfile.gettempdir(), uploaded_subtitle.name)
        with open(subtitle_path, "wb") as f:
            f.write(uploaded_subtitle.read())

    #load distilbert model if subtitles are available
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

    #process video to extract emotions
    with st.spinner("Processing Video..."):
        video_predictions = process_video(temp_video_path, video_model)
        video_emotion_labels = convert_predictions_to_labels(video_predictions, video_emotion_map)

    #free up memory before processing audio
    import gc
    gc.collect()

    #extract and process audio from video
    with st.spinner("Extracting Audio..."):
        video_clip = mp.VideoFileClip(temp_audio_video_path)
        temp_audio_path = temp_video_path.replace(".mp4", ".wav")
        video_clip.audio.write_audiofile(temp_audio_path, verbose=False, logger=None)
        video_duration = video_clip.duration
        video_clip.close()

    with st.spinner("Processing Audio..."):
        audio_predictions = process_audio(temp_audio_path, audio_model)
        audio_emotion_labels = (
            convert_predictions_to_labels(audio_predictions, audio_emotion_map, is_audio=True)
            if audio_predictions is not None and isinstance(audio_predictions, np.ndarray) and audio_predictions.size > 0
            else None
        )

    #analyze text if subtitles are available
    text_predictions = None
    text_emotion_labels = None
    if subtitle_path and text_model and text_tokenizer:
        with st.spinner("Processing Subtitles..."):
            try:
                with open(subtitle_path, "r", encoding="utf-8") as f:
                    subtitles = list(srt.parse(f.read()))
                
                text_segments = [sub.content for sub in subtitles]
                text_predictions = analyze_text_emotion(text_segments, text_model, text_tokenizer)
                text_emotion_labels = convert_predictions_to_labels(text_predictions, text_emotion_map)
                st.success("Subtitle emotion analysis completed!")
            except Exception as e:
                st.error(f"Error processing subtitles: {e}")

    #display results
    st.write("## 📊 Results")
    col1, col2, col3 = st.columns(3)

    #show video emotion predictions
    with col1:
        st.subheader("Video Emotions")
        if video_emotion_labels:
            video_df = pd.DataFrame({'Time (s)': [i*3 for i in range(len(video_emotion_labels))], 'Emotion': video_emotion_labels})
            st.dataframe(video_df)
        else:
            st.warning("No video predictions available.")

    #show audio emotion predictions
    with col2:
        st.subheader("Audio Emotions")
        if audio_emotion_labels:
            audio_df = pd.DataFrame({'Time (s)': [i*3 for i in range(len(audio_emotion_labels))], 'Emotion': audio_emotion_labels})
            st.dataframe(audio_df)
        else:
            st.warning("No audio predictions available.")

    #show text emotion predictions
    with col3:
        st.subheader("Text Emotions")
        if text_emotion_labels:
            text_df = pd.DataFrame({'Text': text_segments, 'Emotion': text_emotion_labels})
            st.dataframe(text_df)
        else:
            st.warning("No text predictions available or text model not loaded.")

    #combine video and audio emotions for multimodal analysis
    if video_emotion_labels and audio_emotion_labels:
        time_points = [i*3 for i in range(max(len(video_predictions), len(audio_predictions)))]
        combined_emotions, combined_probabilities, highest_probabilities = fuse_emotions_bimodal(video_predictions, audio_predictions)
        display_combined_results(st, combined_emotions, highest_probabilities, time_points)

        #plot combined emotion timeline
        st.subheader("Video-Audio Emotion Timeline")
        fig, ax = plt.subplots(figsize=(10, 6))
        all_emotions = ['Neutral', 'Happy', 'Sad', 'Angry', 'Fear', 'Disgust', 'Surprise']
        emotion_to_num = {emotion: i for i, emotion in enumerate(all_emotions)}
        emotion_nums = [emotion_to_num[e] for e in combined_emotions]
        ax.plot(time_points, emotion_nums, 'o-', label='Combined Emotion', color='purple', linewidth=2)
        ax.set_yticks(range(len(all_emotions)))
        ax.set_yticklabels(all_emotions)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Emotion')
        ax.set_title('Video-Audio Emotion Detection Timeline')
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.7)
        st.pyplot(fig)

        #analyze dominant emotions
        emotion_counts = {emotion: combined_emotions.count(emotion) for emotion in set(combined_emotions)}
        dominant_df = pd.DataFrame({'Emotion': list(emotion_counts.keys()), 'Count': list(emotion_counts.values())})
        st.dataframe(dominant_df.sort_values(by='Count', ascending=False))

        st.markdown(f"### Overall Dominant Emotion: {dominant_df.iloc[0]['Emotion']}")
