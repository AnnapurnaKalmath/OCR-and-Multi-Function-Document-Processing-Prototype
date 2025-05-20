import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import io

# Load trained model
model = load_model('model_trained.keras')

# Character mapping: 0–9, A–Z, a–z
class_to_char = {i: str(i) for i in range(10)}
class_to_char.update({i: chr(65 + (i - 10)) for i in range(10, 36)})
class_to_char.update({i: chr(97 + (i - 36)) for i in range(36, 62)})

def preprocess_image(file):
    image = Image.open(file).convert('RGB')
    image = image.resize((128, 128))
    image_array = np.array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    return image_array

def run_ocr(file):
    img = preprocess_image(file)
    prediction = model.predict(img)
    predicted_class = np.argmax(prediction, axis=1)[0]
    predicted_character = class_to_char.get(predicted_class, "Unknown")
    confidence = float(np.max(prediction) * 100)
    return predicted_character, confidence
