from huggingface_hub import login
from transformers import pipeline
import fitz  # PyMuPDF
import tkinter as tk
from tkinter import filedialog

# STEP 1: Login to Hugging Face (safe only for local personal use)
import os

HF_TOKEN = os.getenv("HF_TOKEN")


# STEP 2: Create summarizer pipeline
summarizer = pipeline("summarization", model="facebook/bart-large-cnn") 

# STEP 3: Function to extract PDF text
def extract_text_from_pdf_bytes(file_bytes):
    # file_bytes is a bytes object (from gr.File(type="binary"))
    doc = fitz.open("pdf", file_bytes)
    text = ""
    for page in doc:
        text += page.get_text()
    doc.close()
    return text

# STEP 4: Function to chunk text (since summarization model has token limits)
def chunk_text(text, max_length=1024):
    sentences = text.split('. ')
    chunks = []
    current_chunk = ""

    for sentence in sentences:
        if len(current_chunk) + len(sentence) < max_length:
            current_chunk += sentence + ". "
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence + ". "
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks

# STEP 5: File Picker and summarization
def summarize_document(file_bytes):
    if file_bytes is None:
        return "No file uploaded."
    
    try:
        # file_bytes is bytes, pass directly to fitz
        full_text = extract_text_from_pdf_bytes(file_bytes)
        
        if not full_text.strip():
            return "The document is empty."
        
        chunks = chunk_text(full_text)
        
        summaries = [
            summarizer(chunk, max_length=150, min_length=30, do_sample=False)[0]['summary_text']
            for chunk in chunks
        ]
        
        return "\n\n".join(summaries)
    
    except Exception as e:
        return f"Error: {str(e)}"


