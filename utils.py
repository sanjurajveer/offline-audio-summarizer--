import whisper
from transformers import pipeline, AutoTokenizer
import os
import re
import tempfile
import torch
import logging
import imageio_ffmpeg as ffmpeg

os.environ["PATH"] += os.pathsep + ffmpeg.get_ffmpeg_exe()
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
logging.basicConfig(level=logging.INFO)

def load_transcription_model():
    return whisper.load_model("base")  # 74M parameters

def transcribe_audio(model, audio_file_path, language='en'):
    try:
        result = model.transcribe(audio_file_path, language=language, temperature=0.0)
        return result["text"]
    except Exception as e:
        logging.error(f"Error during transcription: {e}")
        return "[Error transcribing audio]"

def load_summarization_model():
    device = 0 if torch.cuda.is_available() else -1
    summarizer = pipeline("summarization", model="t5-small", device=device)
    tokenizer = AutoTokenizer.from_pretrained("t5-small")
    return summarizer, tokenizer

def clean_text(text):
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()

def chunk_text(text, tokenizer, max_chunk_length=500):
    words = text.split()
    chunks = []
    current_chunk = []

    for word in words:
        current_chunk.append(word)
        if len(tokenizer.encode(' '.join(current_chunk))) > max_chunk_length:
            chunks.append(' '.join(current_chunk[:-1]))
            current_chunk = [word]

    if current_chunk:
        chunks.append(' '.join(current_chunk))

    return chunks

def summarize_long_text(summarizer, tokenizer, text):
    chunks = chunk_text(text, tokenizer, max_chunk_length=500)
    summaries = []
    for chunk in chunks:

        try:
            summary = summarizer(chunk, max_length=150, min_length=50, do_sample=False)[0]['summary_text']
            summaries.append(summary)
        except Exception as e:
            logging.error(f"Error summarizing chunk: {e}")
            summaries.append("[Error summarizing this section]")
    
    return ' '.join(summaries)

def summarize_text(summarizer, tokenizer, text):
    text = clean_text(text)

    if not text.strip():
        return "No content to summarize."
    
    try:
        token_length = len(tokenizer.encode(text, truncation=False, add_special_tokens=False))
        logging.info(f"Input text token length: {token_length}")
        if token_length > 500:
            logging.info("Using long text summarization")
            return summarize_long_text(summarizer, tokenizer, text)
        else:
            logging.info("Using single chunk summarization")
            return summarize_chunk(summarizer, text)
    except Exception as e:
        logging.error(f"Error summarizing text: {e}")
        return "[Error summarizing the text]"

def save_uploaded_file(uploaded_file):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.' + uploaded_file.name.split('.')[-1]) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            return tmp_file.name
    except Exception as e:
        logging.error(f"Error handling uploaded file: {e}")
        return None
