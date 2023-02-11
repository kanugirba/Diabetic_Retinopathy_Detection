import streamlit as st
import os
import re, math
import time
import pandas as pd
import numpy as np
import tensorflow as tf
from utils import preprocess_image, build_model, predict
import cv2
def create_opencv_image_from_stringio(img_stream, cv2_img_flag=0):
    img_stream.seek(0)
    img_array = np.asarray(bytearray(img_stream.read()), dtype=np.uint8)
    return cv2.imdecode(img_array, cv2_img_flag)
model = build_model(ef=4)
model.load_weights("/home/alchemist/Downloads/EfficientnetB4_100Epoch-20230208T152748Z-001/EfficientnetB4_100Epoch/fold-4.h5")
st.write("Diabetic retinopathy is a serious eye condition that can lead to vision loss. Early detection and treatment are crucial for preventing vision loss. However, it is important to note that this software is not a substitute for a professional medical opinion. If you have any concerns about your eye health, please consult a doctor for a comprehensive eye exam and proper diagnosis.")
uploaded_file = st.file_uploader("Upload the Image")
if st.button('Predict'):
    img = create_opencv_image_from_stringio(uploaded_file)
    img = preprocess_image(img, crop=True, blur = True, sigmaX=10)
    st.image(img)
    data = tf.keras.applications.efficientnet.preprocess_input(
        img, data_format=None
    )
    data = np.expand_dims(data, axis=0)
    output , _= predict(model,data)
    classes = ["No","Mild","Moderate","Severe","Proliferative"]
    st.write(f"The uploaded image is having {classes[output[0]]} Diabetic Retinopathy.")


