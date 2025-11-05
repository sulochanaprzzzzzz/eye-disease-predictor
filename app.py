# app.py
import streamlit as st
import numpy as np
from PIL import Image
import os

# --------------------------------------------------------------
# 1. UPDATE WITH YOUR EXACT CLASS FOLDER NAMES
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
# 2. LOAD MODEL (already in repo)
# --------------------------------------------------------------
@st.cache_resource
def load_eye_model():
    model_path = "final_model.keras"
    if not os.path.exists(model_path):
        st.error("`final_model.keras` not found in the repository. "
                 "Make sure it is committed and < 100 MB.")
        return None
    try:
        from tensorflow.keras.models import load_model
        from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
        globals()["preprocess_input"] = preprocess_input
        return load_model(model_path)
    except Exception as e:
        st.error(f"Model load error: {e}")
        return None

model = load_eye_model()
preprocess_input = globals().get("preprocess_input")

# --------------------------------------------------------------
# 3. PREDICTION
# --------------------------------------------------------------
def predict(image_file):
    img = Image.open(image_file).convert("RGB").resize((224, 224))
    x = np.array(img)[np.newaxis, ...]          # (1, 224, 224, 3)
    x = preprocess_input(x)
    preds = model.predict(x)[0]
    idx = int(np.argmax(preds))
    return CLASSES[idx], float(preds[idx])

# --------------------------------------------------------------
# 4. UI
# --------------------------------------------------------------
st.set_page_config(page_title="Eye Disease Detector", page_icon="eyes")
st.title("Eye Disease Prediction")
st.markdown("*Upload a **fundus** eye image – **educational use only**.*")

uploaded = st.file_uploader("Choose image...", type=["jpg", "jpeg", "png"])

if uploaded and model:
    st.image(uploaded, use_column_width=True)
    with st.spinner("Analyzing..."):
        label, conf = predict(uploaded)

    if label == "Normal":
        st.success(f"**{label}** – Confidence: {conf:.1%}")
        st.balloons()
    else:
        st.error(f"**{label.replace('_', ' ')}** – Confidence: {conf:.1%}")
        st.warning("Please consult an ophthalmologist.")
elif uploaded and not model:
    st.error("Model failed to load.")
