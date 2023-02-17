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

model_path = "./fold-4.h5" # replace this by downloading the model file from the drive link attached in the Readme.md file
model = build_model(ef=4)
model.load_weights(model_path)

st.write("Diabetic retinopathy is a serious eye condition that can lead to vision loss. Early detection and treatment are crucial for preventing vision loss. However, it is important to note that this software is not a substitute for a professional medical opinion. If you have any concerns about your eye health, please consult a doctor for a comprehensive eye exam and proper diagnosis.")
uploaded_file = st.file_uploader("Upload the Image")
if st.button('Predict'):
    img = Image.open(uploaded_file)
    img = preprocess_image(np.array(img), crop=True, blur = True, sigmaX=10)
    image = tf.cast(img, tf.float32) / 255.0
    image = tf.reshape(image, [512,512, 3])
    st.image(img)
    data = np.expand_dims(image, axis=0)
    output , proba = predict(model,data)
    classes = ["No","Mild","Moderate","Severe","Proliferative"]
    st.write(f"The uploaded image is having {classes[output[0]]} Diabetic Retinopathy.")
    fig, ax = plt.subplots(figsize =(12, 8))
    ax.barh(range(len(proba[0])),proba[0],align='center')
    plt.yticks(range(len(proba[0])),classes)
    plt.title('Probabilities of Each Class')
    st.pyplot(fig)




