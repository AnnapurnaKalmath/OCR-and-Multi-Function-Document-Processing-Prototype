import gradio as gr
from ocr_model import run_ocr
from keyword_search import process_file
from tts import read_document

import fitz  # PyMuPDF for PDF reading
from transformers import pipeline
import os
import tempfile

# Initialize summarizer pipeline once
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

def extract_text_from_pdf_bytes(pdf_bytes):
    # Open PDF directly from bytes using fitz
    doc = fitz.open("pdf", pdf_bytes)
    text = ""
    for page in doc:
        text += page.get_text()
    doc.close()
    return text

def chunk_text(text, max_length=1024):
    sentences = text.split('. ')
    chunks = []
    current_chunk = ""

    for sentence in sentences:
        # +2 accounts for the ". " being added back
        if len(current_chunk) + len(sentence) + 2 < max_length:
            current_chunk += sentence + ". "
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence + ". "
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks

def summarize_document(file_bytes):
    if file_bytes is None:
        return "No file uploaded."

    try:
        # Extract text directly from bytes
        full_text = extract_text_from_pdf_bytes(file_bytes)

        if not full_text.strip():
            return "The document is empty."

        chunks = chunk_text(full_text)

        summaries = []
        for i, chunk in enumerate(chunks):
            summary = summarizer(chunk, max_length=150, min_length=30, do_sample=False)[0]['summary_text']
            summaries.append(f"--- Summary {i + 1} ---\n{summary}")

        return "\n\n".join(summaries)

    except Exception as e:
        return f"Error: {str(e)}"


# ==== TAB 1: OCR ====
def ocr_from_image(image_file):
    if image_file is None:
        return "No image uploaded."
    try:
        character, confidence = run_ocr(image_file)
        return f"🔤 Predicted Character: {character}\n🎯 Confidence: {confidence:.2f}%"
    except Exception as e:
        return f"Error: {str(e)}"

ocr_tab = gr.Interface(
    fn=ocr_from_image,
    inputs=gr.Image(type="filepath", label="Upload Image"),
    outputs=gr.Textbox(label="OCR Output"),
    title="Image OCR",
    description="Upload an image of a single character. The model will predict it."
)

# ==== TAB 2: Keyword Search ====
keyword_tab = gr.Interface(
    fn=process_file,
    inputs=[
        gr.File(label="Upload scanned PDF", type="binary"),
        gr.Textbox(label="Enter keyword", placeholder="Type keyword to search...")
    ],
    outputs=gr.Textbox(label="Keyword Matches"),
    title="Keyword Search (OCR-Based)",
    description="Uses a custom OCR model to extract text from scanned PDFs and find keyword matches."
)

# ==== TAB 3: Text-to-Speech ====
tts_tab = gr.Interface(
    fn=read_document,
    inputs=gr.File(label="Upload PDF or TXT Document"),
    outputs=[gr.Textbox(label="Status"), gr.Audio(label="Listen to Document")],
    title="Document TTS",
    description="Upload a document and listen to it as audio."
)

# ==== TAB 4: Summarizer ====
summarizer_tab = gr.Interface(
    fn=summarize_document,
    inputs=gr.File(label="Upload PDF or TXT Document", type="binary"),  # 👈 add type="binary"
    outputs=gr.Textbox(label="Summarized Text"),
    title="📄 Summarizer",
    description="Upload a document and get a concise summary using BART transformer model."
)

# ==== Combined Gradio App ====
app = gr.TabbedInterface(
    [ocr_tab, keyword_tab, tts_tab, summarizer_tab],
    ["🧠 OCR", "🔍 Keyword Search", "🔊 Text-to-Speech", "📄 Summarizer"]
)

app.launch(share=True)
