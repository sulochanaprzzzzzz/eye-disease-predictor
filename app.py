import streamlit as st
import numpy as np
from PIL import Image
import os

# Update with your exact class names (from training folders)
CLASSES = ["Cataract", "Diabetic_Retinopathy", "Glaucoma", "Normal", "Other", "Retina_disease"]

# Load model (defer TF imports to avoid early errors)
@st.cache_resource
def load_model_safe():
    model_path = "final_model.keras"
    if not os.path.exists(model_path):
        st.error("Model file 'final_model.keras' not found. Check GitHub commit.")
        return None, None
    
    try:
        # Import TF only when needed
        from tensorflow.keras.models import load_model
        from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
        model = load_model(model_path)
        st.success("Model loaded successfully!")
        return model, preprocess_input
    except ImportError as e:
        st.error(f"TensorFlow not installed: {e}. Check requirements.txt and re-deploy.")
        return None, None
    except Exception as e:
        st.error(f"Model load failed: {e}")
        return None, None

model, preprocess_input = load_model_safe()

def predict_eye_disease(image_file):
    if model is None or preprocess_input is None:
        return None, 0.0
    
    img = Image.open(image_file).convert("RGB").resize((224, 224))
    x = np.array(img)[np.newaxis, ...]  # Shape: (1, 224, 224, 3)
    x = preprocess_input(x)
    predictions = model.predict(x, verbose=0)[0]
    predicted_idx = np.argmax(predictions)
    confidence = predictions[predicted_idx]
    return CLASSES[predicted_idx], confidence

# UI
st.set_page_config(page_title="Eye Disease Detector", page_icon="👁️")
st.title("👁️ Eye Disease Prediction")
st.markdown("**Upload a fundus eye image.** *Educational use only – consult a doctor for real advice.*")

# Status check
if model is None:
    st.error("❌ Model failed to load. See details above.")
    st.stop()

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display image
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
    
    # Predict
    with st.spinner("🔬 Analyzing your eye image..."):
        predicted_label, confidence = predict_eye_disease(uploaded_file)
    
    if predicted_label == "Normal":
        st.success(f"✅ **{predicted_label}** (Confidence: {confidence:.1%})")
        st.balloons()
        st.markdown("Your eyes look healthy! Keep up good habits like regular check-ups.")
    else:
        st.error(f"⚠️ **{predicted_label.replace('_', ' ')}** (Confidence: {confidence:.1%})")
        st.warning("This is **not a diagnosis**. See an ophthalmologist ASAP for professional evaluation.")

st.markdown("---")
st.markdown("*Built with ❤️ using Streamlit & TensorFlow. Repo: [GitHub](https://github.com/sulochanaprzzzzzz/eye-disease-predictor)*")
