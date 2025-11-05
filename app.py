import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from PIL import Image
import os
import gdown

# === UPDATE THESE WITH YOUR ACTUAL CLASSES ===
CLASSES = ['Cataract', 'Diabetic_Retinopathy', 'Glaucoma', 'Normal', 'Other', 'Retina_disease']

@st.cache_resource
def load_eye_model():
    model_path = "final_model.keras"
    if not os.path.exists(model_path):
        st.warning("Model not found. Downloading from Google Drive...")
        # REPLACE WITH YOUR FILE ID
        gdown.download("https://drive.google.com/uc?id=YOUR_FILE_ID_HERE", model_path, quiet=False)
    try:
        return load_model(model_path)
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None

model = load_eye_model()

def predict(image):
    img = Image.open(image).convert('RGB').resize((224, 224))
    x = np.array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    pred = model.predict(x)[0]
    idx = np.argmax(pred)
    return CLASSES[idx], pred[idx]

# === UI ===
st.set_page_config(page_title="Eye Disease Detector", page_icon="eyes")
st.title("Eye Disease Prediction")
st.markdown("*Upload a fundus eye image. For educational use only.*")

uploaded = st.file_uploader("Choose image...", type=["jpg", "jpeg", "png"])

if uploaded and model:
    st.image(uploaded, use_column_width=True)
    with st.spinner("Analyzing..."):
        label, confidence = predict(uploaded)
    if label == "Normal":
        st.success(f"**{label}** (Confidence: {confidence:.1%})")
        st.balloons()
    else:
        st.error(f"**{label.replace('_', ' ')}** (Confidence: {confidence:.1%})")
        st.warning("Consult an eye doctor immediately.")
elif uploaded and not model:
    st.error("Model failed to load.")
