import streamlit as st
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance

st.title("Automated Product Defect Detection")

st.write("Upload a product image to check whether it is defective or non-defective.")

uploaded_file = st.file_uploader("Upload Image", type=["jpg","png","jpeg"])

if uploaded_file is not None:

    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # resize image
    img = img.resize((256,256))

    # convert to grayscale
    gray = img.convert("L")

    # increase contrast
    enhancer = ImageEnhance.Contrast(gray)
    gray = enhancer.enhance(2)

    # detect edges
    edges = gray.filter(ImageFilter.FIND_EDGES)

    edge_array = np.array(edges)

    # calculate edge intensity
    edge_strength = edge_array.mean()

    st.write("Edge Strength:", edge_strength)

    # improved threshold
    if edge_strength > 12:
        result = "Defective Product"
    else:
        result = "Non-Defective Product"

    st.subheader("Prediction Result")
    st.success(result)

    st.subheader("Edge Detection Output")
    st.image(edges, use_column_width=True)
