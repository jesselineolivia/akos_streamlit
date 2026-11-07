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

DEFAULT_LEATHER_CLASSES = [
    "Folding marks",
    "Grain off",
    "Growth marks",
    "Loose grains",
    "Non defective",
    "Pinhole",
]

def get_model_status():
    import zipfile
    model_path = Path(os.getenv("MODEL_PATH", str(DEFAULT_MODEL)))
    if not model_path.exists():
        return {"available": False, "corrupted": False, "path": str(model_path), "error": None}
    try:
        with zipfile.ZipFile(model_path, "r") as z:
            _ = z.namelist()
        return {"available": True, "corrupted": False, "path": str(model_path), "error": None}
    except Exception as e:
        return {"available": False, "corrupted": True, "path": str(model_path), "error": str(e)}

@st.cache_resource
def get_keras_model(model_path_str):
    import tensorflow as tf
    return tf.keras.models.load_model(model_path_str)

def load_classifier():
    import tensorflow as tf
    import zipfile
    model_path = Path(os.getenv("MODEL_PATH", str(DEFAULT_MODEL)))
    labels_path = Path(os.getenv("CLASS_NAMES_PATH", str(DEFAULT_LABELS)))
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    try:
        model = get_keras_model(str(model_path))
    except (zipfile.BadZipFile, Exception) as err:
        if "Bad magic number" in str(err) or isinstance(err, zipfile.BadZipFile):
            raise ValueError(
                f"File model '{model_path.name}' rusak/corrupt (Bad magic number). "
                f"File .keras adalah arsip ZIP yang terpotong saat proses save/download. "
                f"Jalankan ulang cell model.save(...) di notebook untuk membuat ulang filenya."
            ) from err
        raise err

    # Always re-read labels directly so changes to class_names.json apply immediately without cache issues
    class_names = []
    if labels_path.exists():
        try:
            with open(labels_path, "r", encoding="utf-8") as f:
                class_names = json.load(f)
        except Exception:
            class_names = []

    # If class_names is empty or contains numbers like ['0', '1', '2', '3', '4', '5'], use the actual leather classes
    if not class_names or all(str(c).isdigit() for c in class_names):
        class_names = DEFAULT_LEATHER_CLASSES

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

