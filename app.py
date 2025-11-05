# app.py - Streamlit app for Eye Disease Prediction

import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from PIL import Image
import os

# Assuming the 6 classes based on common eye disease datasets (adjust if your classes differ)
# Example: ['Cataract', 'Diabetic Retinopathy', 'Glaucoma', 'Normal', 'Other', 'Retina Disease']
# You need to replace this with your actual class names from the training directory subfolders.
CLASSES = ['Cataract', 'Diabetic Retinopathy', 'Glaucoma', 'Normal', 'Other', 'Retina Disease']

# Load the model (assume the model is in the same directory as app.py)
# If the model is large, consider hosting it on Hugging Face or Google Drive and downloading it on app start.
@st.cache_resource
def load_eye_model():
    model_path = 'final_model.keras'  # Place your model file here
    if not os.path.exists(model_path):
        st.error("Model file not found. Please ensure 'final_model.keras' is in the app directory.")
        return None
    return load_model(model_path)

model = load_eye_model()

def predict_eye_disease(image):
    # Preprocess the image to match training (224x224 for MobileNetV2)
    img = Image.open(image).convert('RGB').resize((224, 224))
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    
    # Make prediction
    predictions = model.predict(img_array)
    predicted_class_idx = np.argmax(predictions, axis=1)[0]
    predicted_label = CLASSES[predicted_class_idx]
    
    return predicted_label

# Streamlit UI
st.title("Eye Disease Prediction")
st.write("Upload an image of an eye to predict if it's normal or detect potential diseases.")
st.write("Note: This is for educational purposes only. Consult a doctor for real medical advice.")

uploaded_file = st.file_uploader("Choose an eye image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
    
    # Predict
    if model is not None:
        with st.spinner("Analyzing..."):
            result = predict_eye_disease(uploaded_file)
        
        if result == 'Normal':
            st.success("Your eyes appear normal!")
        else:
            st.warning(f"Predicted Disease: {result}")
            st.write("Please consult a healthcare professional for confirmation.")
    else:
        st.error("Cannot proceed without the model.")