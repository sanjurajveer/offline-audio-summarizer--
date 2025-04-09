import streamlit as st
import whisper
import requests


# Load Whisper model
whisper_model = whisper.load_model("base")


# Ollama configuration
OLLAMA_HOST = "http://localhost:11434"  # Default Ollama host
LLAMA_MODEL = "llama3"  # Change this to the actual model name you're using in Ollama


def summarize_with_llama(text):
   prompt = f"Summarize the following text:\n\n{text}\n\nSummary:"
   response = requests.post(
       f"{OLLAMA_HOST}/api/generate",
       json={
           "model": LLAMA_MODEL,
           "prompt": prompt,
           "stream": False
       }
   )
   if response.status_code == 200:
       return response.json()['response'].strip()
   else:
       return "Error: Could not generate summary."


st.title("🎙️ Offline Audio Summarizer")


# Upload an audio file
audio_file = st.file_uploader("Upload an audio file", type=["wav", "mp3", "m4a"])


if audio_file is not None:
   # Save uploaded file
   with open("uploaded_audio.mp3", "wb") as f:
       f.write(audio_file.getbuffer())


   st.audio(audio_file, format="audio/mp3")


   # Transcribe using Whisper
   with st.spinner("Transcribing..."):
       transcript = whisper_model.transcribe("uploaded_audio.mp3")["text"]
  
   st.subheader("📜 Transcript")
   st.write(transcript)


   # Summarize using LLaMA3 via Ollama
   with st.spinner("Summarizing..."):
       summary = summarize_with_llama(transcript)


   st.subheader("📝 Summary")
   st.write(summary)
