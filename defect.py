import streamlit as st
import numpy as np
from PIL import Image

st.title("Automated Product Defect Detection")

st.write("Upload a product image to check if it is defective.")

uploaded_file = st.file_uploader("Upload Image", type=["jpg","png","jpeg"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_column_width=True)

    img = img.resize((128,128))
    img_array = np.array(img)

    # simple defect detection logic
    avg_pixel = img_array.mean()

    if avg_pixel < 120:
        result = "Defective Product"
    else:
        result = "Non-Defective Product"

    st.subheader("Prediction Result")
    st.success(result)
