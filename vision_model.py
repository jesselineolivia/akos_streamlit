import json
import os
from pathlib import Path
import numpy as np
import streamlit as st

ROOT = Path(__file__).parent
MODELS_DIR = Path(os.getenv("MODELS_DIR", "D:/KULIAH/Semester 7/M3C/akos_streamlit/models"))
if not MODELS_DIR.exists():
    MODELS_DIR = ROOT / "models"
DEFAULT_MODEL = MODELS_DIR / "resnet50_model.keras"
DEFAULT_LABELS = MODELS_DIR / "class_names.json"

def get_model_status():
    model_path = Path(os.getenv("MODEL_PATH", str(DEFAULT_MODEL)))
    return {"available": model_path.exists(), "path": str(model_path)}

@st.cache_resource
def load_classifier():
    import tensorflow as tf
    model_path = Path(os.getenv("MODEL_PATH", str(DEFAULT_MODEL)))
    labels_path = Path(os.getenv("CLASS_NAMES_PATH", str(DEFAULT_LABELS)))
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    model = tf.keras.models.load_model(model_path)
    if labels_path.exists():
        with open(labels_path, "r", encoding="utf-8") as f:
            class_names = json.load(f)
    else:
        class_names = [str(i) for i in range(int(model.output_shape[-1]))]
    return model, class_names

def predict_image(model, class_names, pil_image):
    import tensorflow as tf
    image = pil_image.resize((224, 224))
    arr = np.asarray(image).astype("float32") / 255.0
    arr = np.expand_dims(arr, axis=0)
    output = model.predict(arr, verbose=0)[0]
    # If the model outputs logits rather than probabilities, convert to softmax.
    if np.any(output < 0) or not np.isclose(float(np.sum(output)), 1.0, atol=0.05):
        output = tf.nn.softmax(output).numpy()
    idx = int(np.argmax(output))
    label = class_names[idx] if idx < len(class_names) else f"Class {idx}"
    probabilities = {class_names[i] if i < len(class_names) else f"Class {i}": float(p)
                     for i, p in enumerate(output)}
    return label, float(output[idx]), probabilities

