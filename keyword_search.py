from pdf2image import convert_from_bytes
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import cv2
import tempfile
import os

# Load the trained OCR model (once, globally)
model = load_model('model_trained.keras')

# Mapping class indices to characters (0-9, A-Z, a-z)
class_to_char = {i: str(i) for i in range(10)}
class_to_char.update({i: chr(65 + (i - 10)) for i in range(10, 36)})
class_to_char.update({i: chr(97 + (i - 36)) for i in range(36, 62)})

def preprocess_image(image):
    """Resize and normalize image for model input."""
    image = image.convert('RGB').resize((128, 128))
    image_array = np.array(image) / 255.0
    return np.expand_dims(image_array, axis=0)

def run_ocr(image):
    """Predict character class from image using the model."""
    img = preprocess_image(image)
    prediction = model.predict(img)
    predicted_class = np.argmax(prediction, axis=1)[0]
    confidence = float(np.max(prediction) * 100)
    return class_to_char.get(predicted_class, "?"), confidence

def convert_pdf_to_images(pdf_bytes):
    """Convert PDF bytes to a list of PIL images (one per page)."""
    images = convert_from_bytes(pdf_bytes)
    return images

def segment_lines_and_characters(pil_image):
    """
    Segment image into lines and characters using contours.
    Returns list of lines, each line a list of bounding boxes.
    """
    img = np.array(pil_image.convert('L'))
    _, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    char_boxes = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w > 5 and h > 5:  # filter noise
            char_boxes.append((x, y, w, h))

    # Group boxes into lines by y coordinate proximity
    char_boxes.sort(key=lambda b: b[1])  # sort top-to-bottom
    lines = []
    line_threshold = 20
    current_line = []
    last_y = None

    for box in char_boxes:
        x, y, w, h = box
        if last_y is None or abs(y - last_y) < line_threshold:
            current_line.append(box)
            last_y = y
        else:
            lines.append(current_line)
            current_line = [box]
            last_y = y
    if current_line:
        lines.append(current_line)

    # Sort characters in each line left-to-right
    lines_sorted = [sorted(line, key=lambda b: b[0]) for line in lines]
    return lines_sorted, pil_image

def extract_text_from_pdf_with_custom_ocr(pdf_bytes):
    """Extract text from PDF bytes using custom OCR model."""
    images = convert_pdf_to_images(pdf_bytes)
    text_data = []

    for page_num, img in enumerate(images, start=1):
        lines, pil_img = segment_lines_and_characters(img)
        for line_num, line_boxes in enumerate(lines, start=1):
            line_text = ""
            for (x, y, w, h) in line_boxes:
                char_img = pil_img.crop((x, y, x + w, y + h))
                char, _ = run_ocr(char_img)
                line_text += char
            text_data.append((page_num, line_num, line_text))

    return text_data

def keyword_search(text_data, keyword):
    """Search for keyword in extracted text, case-insensitive."""
    results = []
    keyword_lower = keyword.lower()
    for page_num, line_num, line in text_data:
        if keyword_lower in line.lower():
            results.append((page_num, line_num, line))
    return results

import fitz  # PyMuPDF

def process_file(file_bytes, keyword):
    try:
        doc = fitz.open("pdf", file_bytes)
        results = []

        if not keyword.strip():
            return "Please enter a valid keyword."

        for page_num, page in enumerate(doc, start=1):  # start=1 for human-friendly page numbers
            lines = page.get_text().split('\n')
            for line_num, line in enumerate(lines, start=1):
                if keyword.lower() in line.lower():
                    result = f"📄 Page {page_num}, 🧾 Line {line_num}: {line}"
                    results.append(result)

        doc.close()

        if results:
            return "\n".join(results)
        else:
            return f"No matches found for keyword: '{keyword}'"
    except Exception as e:
        return f"Error during processing: {str(e)}"



