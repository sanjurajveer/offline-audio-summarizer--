# PodQuik-audio-summarizer
 Podcast (Audio) Summarizer App. <br>
[PodQuik](https://podquik.streamlit.app/)
 
 Uses OpenAI's Whisper model to transcribe audio files and Hugging Face's Transformers to summarize the transcription.

## 🚀 Features <br>
**Transcription:** Leverages OpenAI Whisper's "base" model for accurate speech-to-text conversion.<br>
**Summarization:** Summarizes transcriptions using Hugging Face’s "t5-small" model.<br>
**User-Friendly Interface:** Simple and easy-to-use interface powered by Streamlit.<br>
**Supported Formats:** Supports WAV/MP3 files for input.

## 🔧 Tech Stack <br>
**OpenAI Whisper:** Used for audio transcription.<br>
**Hugging Face Transformers:** Summarization using t5-small model.<br>
**Streamlit:** For building the web application interface.<br>
**PyTorch:** As a backend for running machine learning models.

## 🔍 Usage <br>
<img src="/images/home.png" width="650" height="500"><br>
- Upload an audio file (WAV/MP3 format). <br>
- Wait for the transcription process to complete. <br><br>
<img src="/images/transcription.png" width="650" height="500"><br>
- Click the "Summarize Transcription" button to get a summary of the transcript.
<img src="/images/summary.png" width="650" height="500">
