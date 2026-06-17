import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import json

model = tf.keras.models.load_model("model.keras")

with open("class_names.json","r") as f:
    class_names = json.load(f)

st.title("Rock Paper Scissors Classifier")

uploaded_file = st.file_uploader(
    "Upload Image",
    type=["png","jpg","jpeg"]
)

if uploaded_file:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(image,width=300)

    img = image.resize((224,224))

    img_array = np.array(img)

    img_array = img_array/255.0

    img_array = np.expand_dims(img_array,axis=0)

    prediction = model.predict(img_array)

    index = np.argmax(prediction)

    confidence = np.max(prediction)

    st.success(
        f"Prediction: {class_names[index]}"
    )

    st.write(
        f"Confidence: {confidence*100:.2f}%"
    )
