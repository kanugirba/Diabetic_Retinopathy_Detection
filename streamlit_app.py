import streamlit as st
import os
import re, math
import time
import pandas as pd
import numpy as np
import tensorflow as tf
from utils import preprocess_image, build_model, predict
import cv2
import matplotlib.pyplot as plt
from PIL import Image

# Page configuration
st.set_page_config(
    page_title="Diabetic Retinopathy Detection",
    page_icon="👁️",
    layout="wide"
)

@st.cache_resource
def load_trained_model():
    model_path = "./fold-4.h5"
    if not os.path.exists(model_path):
        import gdown
        st.info("Downloading model weights from Google Drive (71 MB)... Please wait a moment.")
        file_id = "1_x1NjzGzrbdwzWdDHzFk8kkjxZn2ac-i"
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, model_path, quiet=False)
    
    m = build_model(ef=4, weights=None)
    m.load_weights(model_path)
    return m

st.title("👁️ Diabetic Retinopathy Detection")
st.markdown("Automated screening and severity classification using Deep Learning (EfficientNet-B4).")

with st.spinner("Loading AI model..."):
    model = load_trained_model()

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Upload Retinal Scan")
    uploaded_file = st.file_uploader("Upload a fundus/retina image (PNG, JPG, JPEG)", type=["png", "jpg", "jpeg"])
    
    if uploaded_file is not None:
        raw_image = Image.open(uploaded_file).convert("RGB")
        st.image(raw_image, caption="Original Image", use_container_width=True)

with col2:
    st.subheader("Diagnosis & Analysis")
    if uploaded_file is not None:
        if st.button('Run Analysis', type="primary", use_container_width=True):
            with st.spinner("Processing image and predicting..."):
                raw_np = np.array(raw_image)
                processed_img = preprocess_image(raw_np, crop=True, blur=True, sigmaX=10)
                image_tensor = tf.cast(processed_img, tf.float32) / 255.0
                image_tensor = tf.reshape(image_tensor, [512, 512, 3])
                data = np.expand_dims(image_tensor, axis=0)
                
                output, proba = predict(model, data)
                classes = ["No DR", "Mild", "Moderate", "Severe", "Proliferative DR"]
                pred_idx = int(output[0])
                predicted_class = classes[pred_idx]
                confidence = proba[0][pred_idx] * 100
                
                # Display processed image
                st.image(processed_img, caption="Preprocessed Image (Ben Graham's method)", use_container_width=True)
                
                # Severity alerts
                if pred_idx == 0:
                    st.success(f"### Result: **{predicted_class}** (Confidence: {confidence:.2f}%)")
                elif pred_idx in [1, 2]:
                    st.warning(f"### Result: **{predicted_class}** (Confidence: {confidence:.2f}%)")
                else:
                    st.error(f"### Result: **{predicted_class}** (Confidence: {confidence:.2f}%)")
                
                # Probability bar chart
                fig, ax = plt.subplots(figsize=(8, 4))
                y_pos = range(len(classes))
                probs = proba[0]
                bars = ax.barh(y_pos, probs, color=['#2ecc71', '#f39c12', '#e67e22', '#e74c3c', '#c0392b'])
                ax.set_yticks(y_pos)
                ax.set_yticklabels(classes, fontsize=11)
                ax.set_xlabel('Probability', fontsize=11)
                ax.set_title('Class Probability Distribution', fontsize=13, fontweight='bold')
                ax.set_xlim(0, 1.0)
                for bar in bars:
                    width = bar.get_width()
                    ax.text(width + 0.02, bar.get_y() + bar.get_height()/2, f'{width*100:.1f}%', 
                            va='center', ha='left', fontsize=10)
                plt.tight_layout()
                st.pyplot(fig)
    else:
        st.info("👆 Please upload a retinal fundus image on the left to begin diagnosis.")

st.divider()
st.caption("⚠️ **Medical Disclaimer**: Diabetic retinopathy is a serious eye condition that can lead to vision loss. Early detection and treatment are crucial. This AI tool is designed for research/screening assistance and is not a substitute for a professional medical diagnosis by an ophthalmologist.")
