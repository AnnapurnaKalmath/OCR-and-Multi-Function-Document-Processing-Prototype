# 🧠 OCR & Document Search Web Application

A powerful **AI-based document analysis tool** with a **custom-trained OCR engine (ResNet CNN)** that supports **keyword search**, **text summarization**, and **text-to-speech (TTS)** — all accessible via an intuitive **Gradio interface**. No third-party OCRs used — *we built our own brain 🧠!*

---

## 🔍 Features

### 🧠 Custom OCR (ResNet CNN-based)
- PDF pages are converted to images using `pdf2image`.
- Images undergo preprocessing: binarization, thresholding, and character blob detection via OpenCV.
- Each character is cropped and passed to a **CNN model trained from scratch** on a dataset combining `0-9`, `A-Z`, and `a-z` (total ~21,000 images).
- Returns accurate text predictions — character-by-character and reconstructed line-by-line.

### 🔍 Keyword Search
- Implements a simple but effective Python logic to search user-given keywords.
- Only uses the **predicted text from the custom OCR model** (no external OCR like Tesseract or PaddleOCR used).

### 🧠 Summarization
- Employs **Facebook's BART (via Hugging Face pipeline)** to summarize large blocks of OCR-extracted text.
- Handles long documents by **chunking** them appropriately to stay within BART's token limit.

### 🔊 Text-to-Speech (TTS)
- Converts extracted/summarized text to **speech** using `gTTS` (Google Text-to-Speech).
- Allows users to **listen** to content on the fly — like an audiobook for your documents!

---

## ⚙️ Tech Stack

| Feature            | Library / Model Used            |
|--------------------|----------------------------------|
| OCR Model          | Custom CNN (ResNet-like)         |
| Image Preprocessing| OpenCV, NumPy                    |
| PDF → Image        | `pdf2image`                      |
| Summarization      | Facebook BART (`transformers`)   |
| TTS                | gTTS                             |
| UI Interface       | Gradio                           |
| Dataset            | 0-9, A-Z, a-z from multiple sources (21k images total) |

---


## 📁 Project Structure

```bash
.
├── main/
│
├── ocr_webapp/
│   ├── app.py              # Gradio interface
│   ├── ocr_model.py        # Custom OCR model loading & inference
│   ├── keyword_search.py   # Keyword search logic
│   ├── summarizer.py       # Text summarization using BART
│   ├── tts.py              # Text-to-Speech using gTTS
│
├── ocr_training/
│   └── ocr_training.ipynb  # CNN (ResNet-style) training notebook
│
├── dataset/
│   └── (0-9, A-Z, a-z)     # 21K labeled character images

---

## 🔗 Skip Connections in ResNet
Skip connections are the core innovation in ResNet that allow us to train very deep networks.

🧠 Why Skip Connections?
In traditional CNNs, deeper layers lead to vanishing gradients. ResNet solves this using residual connections:

Instead of just passing the transformed output forward, we add the original input back in:
