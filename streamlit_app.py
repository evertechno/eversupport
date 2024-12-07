import streamlit as st
import whisper
from transformers import pipeline
from pydub import AudioSegment
import os

# Title and Instructions
st.title("Open-Source Audio Call Analyzer")
st.write("Upload an audio file, get a transcription, and analyze it using AI.")

# File uploader for audio files
uploaded_file = st.file_uploader("Upload your audio file (MP3/WAV):", type=["mp3", "wav"])

if uploaded_file is not None:
    # Save the uploaded file to a temporary location
    temp_file_path = f"temp_audio.{uploaded_file.name.split('.')[-1]}"
    with open(temp_file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.success(f"Uploaded: {uploaded_file.name}")

    # Convert to WAV if necessary
    if uploaded_file.name.endswith(".mp3"):
        audio = AudioSegment.from_mp3(temp_file_path)
        temp_wav_path = "temp_audio.wav"
        audio.export(temp_wav_path, format="wav")
        temp_file_path = temp_wav_path

    # Transcribe the audio using Whisper
    st.write("Transcribing audio...")
    model = whisper.load_model("base")  # Whisper model size: tiny, base, small, medium, large
    transcription_result = model.transcribe(temp_file_path)
    transcription = transcription_result["text"]

    # Display transcription
    st.success("Transcription completed!")
    st.text_area("Transcription:", transcription, height=200)

    # Remove temporary audio file
    os.remove(temp_file_path)

    # Analyze transcription using Hugging Face Transformers
    if st.button("Analyze with AI"):
        st.write("Analyzing transcription...")

        # Summarization pipeline
        summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
        summary = summarizer(transcription, max_length=130, min_length=30, do_sample=False)

        # Display the summary
        st.write("AI Summary:")
        st.write(summary[0]["summary_text"])
