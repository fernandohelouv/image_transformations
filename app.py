"""
    Image Transformations

    This app allows you to apply different transformations to images.
"""

import os
from matplotlib.style import available
import streamlit as st
from PIL import Image
from transformations import (
    get_histogram,
    enhance_contrast,
    grayscale,
    umbral,
    invert,
    equalization,
    media_filter,
    median_filter,
    apply_prewitt,
    apply_kirsch,
    apply_sobel,
    cany,
)


# Get the list of images
images = os.listdir("./input")
images.sort()
images.insert(0, "Select an image")

# Title
st.title("Image Transformations 🌋")

st.markdown("---")

# Create a list of the available transformation options and map them to their respective functions
transformation_options = {
    "Select a Transformation": None,
    "Enhance Contrast": enhance_contrast,
    # "histogram": get_histogram,  #
    "Turn to Gray Scale": grayscale,
    "Umbral Binarization": umbral,
    "Invert Color": invert,
    "Luminance Equalization": equalization,
    "Noise Filter (Media)": media_filter,
    "Noise Filter (Median)": median_filter,
    "Edge Detection (Prewitt)": apply_prewitt,
    "Edge Detection (Kirsch)": apply_kirsch,
    "Edge Detection (Sobel)": apply_sobel,
    "Edge Detection (Canny)": cany,
}

colOriginal, colTransformed = st.columns(2)

img = None

with colOriginal:
    # Field for uploading an image
    uploaded_file = st.file_uploader("Upload an image for transformation")
    selected_image = st.selectbox("Or elect an example image from the gallery", images)

    st.subheader("Original Image")

    # Create a selectbox for image selection
    if selected_image:
        if selected_image != "Select an image":
            st.image(f"./input/{selected_image}", caption=selected_image)
            img = Image.open(f"./input/{selected_image}")

    if uploaded_file is not None:
        st.image(uploaded_file, caption="Uploaded Image")
        img = Image.open(uploaded_file)

    if img is not None:
        original_histogram = get_histogram(img)
        st.bar_chart(original_histogram)

with colTransformed:
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    selected_transformation = st.selectbox(
        "Select a transformation", transformation_options.keys()
    )

    if selected_transformation == "Umbral Binarization":
        umbral_value = st.slider("Umbral value", 0, 255, 127)

    tButton = None

    if img is None:
        tButton = st.button(
            "Transform", type="primary", use_container_width=True, disabled=True
        )
    elif img is not None and selected_transformation == "Select a Transformation":
        tButton = st.button(
            "Transform", type="primary", use_container_width=True, disabled=True
        )
    else:
        tButton = st.button("Transform", type="primary", use_container_width=True)

    st.subheader("Transformed Image")
    # Apply the selected image transformation algorithm
    if tButton:
        # Get the selected transformation
        transformation = transformation_options[selected_transformation]
        transformed_img = transformation(img)

        if isinstance(transformed_img, dict):
            for key, value in transformed_img.items():
                st.image(value, caption=f"Transformed Image {key}")
        else:
            st.image(transformed_img, caption="Transformed Image")

            transformed_histogram = get_histogram(transformed_img)
            st.bar_chart(transformed_histogram)

# * Image Gallery
# Create a gallery with all the images from the input folder
st.sidebar.subheader("Image Gallery")
# Create a list to hold the columns
columns = []

# Create enough columns to hold all images
for i in range(0, len(images), 4):
    columns.extend(st.columns(4))

available_images = os.listdir("./input")
available_images.sort()
# Display each image in a column
for i, image in enumerate(available_images):
    with columns[i]:
        st.sidebar.image(f"./input/{image}", caption=image)
