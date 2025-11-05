# ──────────────────────────────────────────────────────────────
# app.py  –  Eye‑Disease Predictor (Streamlit Cloud ready)
# ──────────────────────────────────────────────────────────────
import os
import streamlit as st
import numpy as np
from PIL import Image

# --------------------------------------------------------------
# 1. Silence TF warnings (optional but clean)
# --------------------------------------------------------------
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# --------------------------------------------------------------
# 2. CLASS NAMES – **EXACTLY the folder names you trained on**
# --------------------------------------------------------------
CLASSES = [
    "Cataract",
    "Diabetic_Retinopathy",
    "Glaucoma",
    "Normal",
    "Other",
    "Retina_disease"
]

# --------------------------------------------------------------
# 3. Load model – TF imports happen *inside* the function
# --------------------------------------------------------------
@st.cache_resource
def get_model():
    path = "final_model.keras"
    if not os.path.exists(path):
        st.error("`final_model.keras` is missing from the repo.")
        return None, None

    try:
        from tensorflow.keras.models import load_model
        from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
        mdl = load_model(path)
        st.success("Model loaded!")
        return mdl, preprocess_input
    except Exception as e:
        st.error(f"Failed to load model: **{e}**")
        return None, None

model, preprocess = get_model()

# --------------------------------------------------------------
# 4. Prediction
# --------------------------------------------------------------
def predict(img_file):
    if model is None:
        return None, 0.0
    img = Image.open(img_file).convert("RGB").resize((224, 224))
    x = np.expand_dims(np.array(img), axis=0)
    x = preprocess(x)
    probs = model.predict(x, verbose=0)[0]
    idx = int(np.argmax(probs))
    return CLASSES[idx], float(probs[idx])

# --------------------------------------------------------------
# 5. UI
# --------------------------------------------------------------
st.set_page_config(page_title="Eye Disease Detector", page_icon="eyes")
st.title("eyes Eye Disease Prediction")
st.caption("Upload a **fundus** image – *educational demo only*")

if model is None:
    st.stop()

file = st.file_uploader("Choose JPG/PNG", type=["jpg", "jpeg", "png"])
if file:
    st.image(file, use_column_width=True)
    with st.spinner("Analyzing…"):
        label, conf = predict(file)

    if label == "Normal":
        st.success(f"**{label}** – {conf:.1%} confidence")
        st.balloons()
    else:
        st.error(f"**{label.replace('_', ' ')}** – {conf:.1%} confidence")
        st.warning("Consult an ophthalmologist – this is **not** a diagnosis.")
