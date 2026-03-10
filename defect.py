import streamlit as st
import numpy as np
from PIL import Image, ImageFilter

# Title
st.title("Automated Product Defect Detection")

st.write("Upload a product image to detect whether it is defective or non-defective.")

# Upload image
uploaded_file = st.file_uploader("Upload Product Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:

    # Open image
    img = Image.open(uploaded_file)

    # Show image
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Convert to grayscale
    img_gray = img.convert("L")

    # Detect edges (helps identify cracks or broken areas)
    edges = img_gray.filter(ImageFilter.FIND_EDGES)

    # Convert to numpy array
    edge_array = np.array(edges)

    # Calculate edge intensity
    edge_strength = edge_array.mean()

    # Defect detection logic
    if edge_strength > 25:
        result = "Defective Product"
    else:
        result = "Non-Defective Product"

    # Show result
    st.subheader("Prediction Result")
    st.success(result)

    # Show edge image for visualization
    st.subheader("Edge Detection")
    st.image(edges, caption="Detected Edges", use_column_width=True)
