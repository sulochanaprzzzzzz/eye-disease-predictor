# app.py
import os
import streamlit as st
import numpy as np
from PIL import Image

# --------------------------------------------------------------
# 1. Suppress TensorFlow warnings (optional but clean)
# --------------------------------------------------------------
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"   # 0=all, 1=INFO, 2=WARNING, 3=ERROR

# --------------------------------------------------------------
# 2. UPDATE THIS LIST WITH YOUR **exact** training folder names
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
# 3. Load model – TensorFlow imports happen *inside* the function
# --------------------------------------------------------------
@st.cache_resource
def load_model_safe():
    model_path = "final_model.keras"

    if not os.path.exists(model_path):
        st.error(
            "`final_model.keras` not found in the repository. "
            "Make sure it is committed and < 100 MB."
        )
        return None, None

    try:
        from tensorflow.keras.models import load_model
        from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

        model = load_model(model_path)
        st.success("Model loaded successfully!")
        return model, preprocess_input

    except ImportError as e:
        st.error(
            f"TensorFlow could not be imported: **{e}**  \n"
            "Check `requirements.txt` and that the app is using **Python 3.11** (`runtime.txt`)."
        )
        return None, None
    except Exception as e:
        st.error(f"Model failed to load: **{e}**")
        return None, None


model, preprocess_input = load_model_safe()


# --------------------------------------------------------------
# 4. Prediction function
# --------------------------------------------------------------
def predict_eye_disease(image_file):
    if model is None or preprocess_input is None:
        return None, 0.0

    img = Image.open(image_file).convert("RGB").resize((224, 224))
    x = np.array(img)[np.newaxis, ...]           # (1, 224, 224, 3)
    x = preprocess_input(x)

    preds = model.predict(x, verbose=0)[0]       # (6,)
    idx = int(np.argmax(preds))
    confidence = float(preds[idx])
    return CLASSES[idx], confidence


# --------------------------------------------------------------
# 5. Streamlit UI
# --------------------------------------------------------------
st.set_page_config(
    page_title="Eye Disease Detector",
    page_icon="eyes",
    layout="centered"
)

st.title("eyes Eye Disease Prediction")
st.markdown(
    """
    **Upload a fundus eye image.**  
    *Educational / demo use only – always consult an ophthalmologist for a real diagnosis.*
    """
)

# ------------------------------------------------------------------
# Stop the app early if the model didn't load (user sees a clear msg)
# ------------------------------------------------------------------
if model is None:
    st.error("Model failed to load – see the messages above.")
    st.stop()

uploaded = st.file_uploader(
    "Choose an image (JPG / PNG)",
    type=["jpg", "jpeg", "png"]
)

if uploaded is not None:
    # Show the image
    st.image(uploaded, caption="Uploaded Image", use_column_width=True)

    # Run prediction
    with st.spinner("Analyzing the eye image..."):
        label, conf = predict_eye_disease(uploaded)

    # ------------------------------------------------------------------
    # Result display
    # ------------------------------------------------------------------
    if label == "Normal":
        st.success(f"**{label}** – Confidence: **{conf:.1%}**")
        st.balloons()
        st.markdown("Your eye looks **healthy** in this image! Keep regular check-ups.")
    else:
        pretty_label = label.replace("_", " ")
        st.error(f"**{pretty_label}** – Confidence: **{conf:.1%}**")
        st.warning(
            "This is **not a medical diagnosis**. "
            "Please see an eye specialist as soon as possible."
        )

# ------------------------------------------------------------------
# Footer
# ------------------------------------------------------------------
st.markdown("---")
st.markdown(
    "*Built with Streamlit & TensorFlow • "
    "[GitHub repo](https://github.com/sulochanaprzzzzzz/eye-disease-predictor)*"
)
