import fitz  # PyMuPDF for PDF processing
from gtts import gTTS
import tempfile
import os

def extract_text_from_pdf(pdf_file):
    """Extracts full text from a PDF file."""
    doc = fitz.open(pdf_file.name)
    text = ""
    for page in doc:
        text += page.get_text("text") + "\n"
    return text

def read_document(file):
    """Reads a full document (PDF or TXT) and converts it to speech."""
    if file is None:
        return "No file uploaded.", None

    if file.name.endswith(".pdf"):
        text = extract_text_from_pdf(file)
    else:
        text = file.read().decode("utf-8")

    if not text.strip():
        return "The document is empty.", None

    # Convert text to speech using gTTS and save to temp file
    tts = gTTS(text)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_audio:
        tts.save(tmp_audio.name)
        return "🎧 Document successfully converted to speech!", tmp_audio.name
