import cv2
import numpy as np
import pyaudio
import wave
import torch
import librosa
import threading
import streamlit as st
from deepface import DeepFace
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

# Load pre-trained models
speech_model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")

# Audio Recording Settings
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK = 1024
RECORD_SECONDS = 5
AUDIO_FILENAME = "output.wav"

# Streamlit UI
st.set_page_config(page_title="Speech & Emotion Recognition", layout="centered")
st.title("🔥 Speech and Emotion Recognition System 🔥")
st.markdown("---")

# Create two tabs
tab1, tab2 = st.tabs(["🎭 Emotion Recognition", "🗣️ Speech Recognition"])

# Function to recognize emotion from frame
def recognize_emotion(frame):
    try:
        result = DeepFace.analyze(frame, actions=["emotion"], enforce_detection=False)
        return result[0]['dominant_emotion']
    except:
        return "Unknown"

# Function to recognize speech from the recorded audio file
def recognize_speech(audio_path):
    try:
        speech, sr = librosa.load(audio_path, sr=16000)
        input_values = processor(speech, return_tensors="pt", sampling_rate=sr).input_values
        logits = speech_model(input_values).logits
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = processor.batch_decode(predicted_ids)[0]
        return transcription
    except Exception as e:
        st.error(f"Speech Recognition Error: {e}")
        return "Error"

# Function to record audio
def record_audio():
    audio = pyaudio.PyAudio()
    stream = audio.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
    
    st.write("🔴 Recording Audio... Speak now!")
    frames = [stream.read(CHUNK) for _ in range(int(RATE / CHUNK * RECORD_SECONDS))]
    st.write("✅ Finished Recording")

    stream.stop_stream()
    stream.close()
    audio.terminate()

    # Save recorded audio
    with wave.open(AUDIO_FILENAME, 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(audio.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))

# Emotion Recognition Tab
with tab1:
    st.header("🎭 Emotion Recognition")
    start_emotion = st.button("▶️ Start Emotion Recognition")
    stop_emotion = st.button("⏹️ Stop Emotion Recognition")

    if start_emotion:
        cap = cv2.VideoCapture(0)
        frame_placeholder = st.empty()
        emotion_placeholder = st.empty()
        stop_flag = False
        
        while not stop_flag:
            ret, frame = cap.read()
            if not ret:
                break
            
            emotion = recognize_emotion(frame)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_placeholder.image(frame_rgb, use_container_width=True)

            # Styled emotion text
            emotion_html = f"""
                <div style='text-align: center; padding: 10px;'>
                    <span style='font-size: 32px; font-weight: bold; color: #FF4B4B;'>
                        Emotion Detected: {emotion.upper()}
                    </span>
                </div>
            """
            emotion_placeholder.markdown(emotion_html, unsafe_allow_html=True)

            if stop_emotion:
                stop_flag = True
                break
        
        cap.release()
        cv2.destroyAllWindows()

# Speech Recognition Tab
with tab2:
    st.header("🗣️ Speech Recognition")
    start_speech = st.button("🎙️ Start Speech Recognition")
    
    if start_speech:
        record_audio()
        speech_text = recognize_speech(AUDIO_FILENAME)
        st.subheader("✅ Recognized Speech")
        st.markdown(f"<p style='font-size:22px; color:#1F618D;'><b>{speech_text}</b></p>", unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("<h4 style='text-align: center;'>Made by <span style='color:#FF5733;'>Sumeet Pandey</span> ❤️</h4>", unsafe_allow_html=True)
