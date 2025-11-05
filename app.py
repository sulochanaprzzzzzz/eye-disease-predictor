import streamlit as st
import numpy as np
from PIL import Image

# Update with your actual class names
CLASSES = ["Cataract", "Diabetic_Retinopathy", "Glaucoma", "Normal", "Other", "Retina_disease"]

@st.cache_resource
def load_model_safe():
    try:
        from tensorflow.keras.models import load_model
        from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
        model = load_model("final_model.keras")
        return model, preprocess_input
    except Exception as e:
        st.error(f"Model load error: {e}")
        return None, None

model, preprocess_input = load_model_safe()

def predict(img_file):
    img = Image.open(img_file).convert("RGB").resize((224, 224))
    x = np.array(img)[np.newaxis, ...]
    x = preprocess_input(x)
    pred = model.predict(x)[0]
    idx = np.argmax(pred)
    return CLASSES[idx], pred[idx]

# UI
st.title("Eye Disease Prediction")
st.markdown("*Upload a fundus eye image – educational use only.*")

uploaded = st.file_uploader("Choose image...", type=["jpg", "jpeg", "png"])

if uploaded and model:
    st.image(uploaded, use_column_width=True)
    with st.spinner("Analyzing..."):
        label, conf = predict(uploaded)
    if label == "Normal":
        st.success(f"**{label}** ({conf:.1%})")
        st.balloons()
    else:
        st.error(f"**{label.replace('_', ' ')}** ({conf:.1%})")
        st.warning("Consult a doctor.")
elif uploaded:
    st.error("Model failed to load.")
