import streamlit as st
from utils import load_transcription_model, transcribe_audio, load_summarization_model, summarize_text, save_uploaded_file

transcription_model = load_transcription_model()
summarization_model, tokenizer = load_summarization_model()

st.set_page_config(page_title="PodQuik", page_icon="🎙️", layout="centered")

st.title('🎙️ PodQuik - Audio Summarizer')
st.markdown("""
    <style>
    .main {background-color: #f0f2f6; color: #333;}
    .css-18e3th9 {padding-top: 2rem; padding-bottom: 2rem;}
    .css-1d391kg {background-color: #00aaff; padding: 5px; border-radius: 5px; text-align: center;}
    .css-1r6slb0 {color: #333; background-color: white;}
    </style>
""", unsafe_allow_html=True)

with st.expander("ℹ️  About this app"):
    st.write("""
        This app uses OpenAI's **Whisper** model to transcribe audio files and Hugging Face's **Transformers** to summarize the transcription.
        \nUpload your audio file in **WAV** or **MP3** format and get a concise summary of its content.
    """)

st.markdown("### Upload your audio file 🎶")
audio_file = st.file_uploader("Upload an audio file", type=['wav', 'mp3'])

if audio_file is not None:
    audio_path = save_uploaded_file(audio_file)
    st.audio(audio_path, format='audio/wav')

    with st.spinner('Transcribing your audio...'):
        transcript = transcribe_audio(transcription_model, audio_path)
        st.markdown("### 📜 Transcription")
        st.write(transcript)

    if st.button('Summarize Transcription 📝'):
        with st.spinner('Summarizing the transcription...'):
            summary = summarize_text(summarization_model, tokenizer, transcript)
            st.markdown("### 📝 Summary")
            st.info(summary)

st.markdown("""
    <hr style="margin-top: 2rem;">
    <div style="text-align: center;">
        Created by <a href="https://github.com/ab490" target="_blank">ab490</a>
    </div>
""", unsafe_allow_html=True)
