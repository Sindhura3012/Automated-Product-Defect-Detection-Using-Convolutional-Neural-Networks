import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image

st.title("Automated Product Defect Detection Using CNN")

st.write("Upload a product image to check whether it is defective or non-defective.")

# Load trained model
@st.cache_resource
def load_cnn_model():
    model = load_model("model.h5")
    return model

model = load_cnn_model()

# Image uploader
uploaded_file = st.file_uploader("Upload Product Image", type=["jpg","png","jpeg"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_column_width=True)

    img = img.resize((128,128))
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)

    if prediction[0][0] > 0.5:
        result = "Defective Product"
    else:
        result = "Non-Defective Product"

    st.subheader("Prediction Result")
    st.success(result)
