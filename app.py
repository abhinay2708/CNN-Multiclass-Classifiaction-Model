import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from tensorflow.keras.models import load_model

# Load model
model = load_model("cifar10_cnn.h5")

# CIFAR-10 class names
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

st.title("🚀 CIFAR-10 Image Classifier")
st.write('''Upload an image and the model will predict its class.
         (Please upload 'airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'.) ''')

# Upload file
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Read file
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)  # 1 = color image
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Resize for CIFAR-10 (32x32)
    img_resized = cv2.resize(img_rgb, (32,32)) / 255.0
    img_array = np.expand_dims(img_resized, axis=0)

    # Prediction
    pred = np.argmax(model.predict(img_array))
    st.image(img_rgb, caption=f"Prediction: {class_names[pred]}", use_container_width=True)

