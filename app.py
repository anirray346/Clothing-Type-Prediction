import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import json
import os

# --- Page Configuration ---
st.set_page_config(
    page_title="Clothing Type Predictor",
    page_icon="👕",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# --- Custom Premium Light Mode CSS ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
        background-color: #f8f9fa;
        color: #212529;
    }
    
    /* Main container styling for glass/card effect */
    .block-container {
        background-color: #ffffff;
        padding: 3rem;
        border-radius: 20px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.05);
        margin-top: 2rem;
        margin-bottom: 2rem;
        border: 1px solid #e9ecef;
    }

    h1, h2, h3 {
        color: #1a1a1a;
        font-weight: 700;
        letter-spacing: -0.5px;
    }
    
    .title-gradient {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 5rem;
        font-weight: 800;
        margin-bottom: 0.5rem;
        text-align: center;
    }

    /* Uploader styling */
    .stFileUploader > div > div {
        background-color: #f8f9fa;
        border: 2px dashed #ced4da;
        border-radius: 15px;
        transition: all 0.3s ease;
    }
    .stFileUploader > div > div:hover {
        border-color: #4facfe;
        background-color: #f1f8ff;
    }

    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        font-size: 1.1rem;
        transition: transform 0.2s, box-shadow 0.2s;
        width: 100%;
        box-shadow: 0 4px 15px rgba(79, 172, 254, 0.4);
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(79, 172, 254, 0.6);
        color: white;
    }

    /* Result box styling */
    .result-box {
        background-color: #f8f9fa;
        padding: 2rem;
        border-radius: 15px;
        border-left: 5px solid #4facfe;
        margin-top: 2rem;
    }
    .result-label {
        font-size: 2rem;
        font-weight: 700;
        color: #212529;
        margin-bottom: 0.5rem;
    }
    .result-confidence {
        font-size: 1.2rem;
        color: #6c757d;
    }
    
    /* Progress bar override */
    .stProgress > div > div > div {
        background-color: #4facfe;
    }
    </style>
""", unsafe_allow_html=True)

# --- Caching the Model and Classes ---
@st.cache_resource(show_spinner="Loading Deep Learning Model...")
def load_model_and_classes():
    model_path = 'clothing_model.h5'
    classes_path = 'classes.json'
    
    if not os.path.exists(model_path) or not os.path.exists(classes_path):
        return None, None
        
    model = tf.keras.models.load_model(model_path)
    with open(classes_path, 'r') as f:
        class_indices = json.load(f)
        
    # Reverse the dictionary to get index -> class name mapping
    labels = {v: k for k, v in class_indices.items()}
    return model, labels

model, labels = load_model_and_classes()

# --- App UI ---
st.markdown('<h1 class="title-gradient" style="font-size: 5rem !important; text-align: center;">Clothing Type Predictor</h1>', unsafe_allow_html=True)
st.markdown("<p style='font-size: 1.1rem; color: #6c757d; margin-bottom: 2rem; text-align: center;'>Upload an image of a clothing item, and our Computer Vision model will automatically identify what type of clothing it is.</p>", unsafe_allow_html=True)

if model is None:
    st.warning("⚠️ Model files not found! Please run the `clothing_prediction.ipynb` notebook completely first to generate and save `clothing_model.h5` and `classes.json`.")
else:
    # File Uploader
    uploaded_file = st.file_uploader("Drag and drop your image here", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("<h3 style='font-size: 1.2rem; margin-bottom: 1rem;'>Uploaded Image:</h3>", unsafe_allow_html=True)
            # Display uploaded image with styling
            image = Image.open(uploaded_file).convert('RGB')
            st.image(image, use_container_width=True, clamp=True)
            
        with col2:
            st.markdown("<h3 style='font-size: 1.2rem; margin-bottom: 1rem;'>Analysis:</h3>", unsafe_allow_html=True)
            analyze_button = st.button("Analyze Image")
            
            if analyze_button:
                with st.spinner("Processing image..."):
                    # Preprocess exactly as in the notebook
                    img_resized = image.resize((224, 224), Image.Resampling.LANCZOS)
                    img_array = tf.keras.utils.img_to_array(img_resized) / 255.0
                    img_array = np.expand_dims(img_array, axis=0)
                    
                    # Predict
                    predictions = model.predict(img_array)
                    predicted_idx = np.argmax(predictions)
                    predicted_class = labels[predicted_idx]
                    confidence = float(predictions[0][predicted_idx])
                    
                    # Display Result
                    st.markdown(f"""
                        <div class="result-box">
                            <div class="result-label">{predicted_class}</div>
                            <div class="result-confidence">Confidence: {confidence*100:.1f}%</div>
                        </div>
                    """, unsafe_allow_html=True)
                    
                    st.progress(confidence)
