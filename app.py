"""
# My first app
Here's our first attempt at using data to create a table:
"""

import os
import streamlit as st
from PIL import Image
from main import (
    enhance_contrast,
    get_histogram,
    get_histogram_from_file,
    turn_to_grayscale,
    apply_umbral_transformation,
    apply_inverse_transformation,
)
from equalization import equalization
from conv_filters import conv_filter, median_filter
from kernels import apply_prewitt, apply_kirsch, apply_sobel
from edge_detection_cv import cany

# Get the list of images
images = os.listdir("./input")
images.sort()

# Title
st.title("Image Transformations 🗻 => 🌋")

# Create a list of the available transformation options and map them to their respective functions
transformation_options = {
    "contrast": enhance_contrast,  # ✅
    # "histogram": get_histogram,  #
    "grayscale": turn_to_grayscale,  # ✅
    "umbral": apply_umbral_transformation,  # ✅
    "inverse": apply_inverse_transformation,  # ✅
    "equalization": equalization,  # ✅
    "convolution": conv_filter,  # ✅
    "median": median_filter,  # ✅
    "prewitt": apply_prewitt,  # ✅
    "kirsch": apply_kirsch,  # ✅
    "sobel": apply_sobel,  # ✅
    "canny": cany,  # ✅
}

# Create a selectbox for image selection
selected_image = st.selectbox("Select an image for transformation", images)
selected_transformation = st.selectbox(
    "Select a transformation", transformation_options.keys()
)

if selected_transformation.lower() == "umbral":
    umbral_value = st.slider("Umbral value", 0, 255, 127)

# Load the selected image
img = Image.open(f"./input/{selected_image}")

# Apply the selected image transformation algorithm
if st.button("Transform"):
    # Get the selected transformation
    transformation = transformation_options[selected_transformation]

    if selected_transformation.lower() == "convolution":
        transformed_img = transformation(f"./input/{selected_image}")
        # Create two columns
        col1, col2 = st.columns(2)

        # Display the original image in the first column
        with col1:
            st.image(img, caption="Original Image")

        # Display the transformed image in the second column
        with col2:
            st.image(transformed_img, caption="Transformed Image")

    elif selected_transformation.lower() == "median":
        transformed_img = transformation(f"./input/{selected_image}")

        # Create two columns
        col1, col2 = st.columns(2)

        # Display the original image in the first column
        with col1:
            st.image(img, caption="Original Image")

        # Display the transformed image in the second column
        with col2:
            st.image(transformed_img, caption="Transformed Image")

    elif selected_transformation.lower() == "prewitt":
        transformed_img = transformation(f"./input/{selected_image}")

        # Create two columns
        col1, col2 = st.columns(2)

        # Display the original image in the first column
        with col1:
            st.image(img, caption="Original Image")

        # Display the transformed image in the second column
        with col2:
            for key, value in transformed_img.items():
                st.image(value, caption=f"Transformed Image {key}")

    elif selected_transformation.lower() == "kirsch":
        transformed_img = transformation(f"./input/{selected_image}")

        # Create two columns
        col1, col2 = st.columns(2)

        # Display the original image in the first column
        with col1:
            st.image(img, caption="Original Image")

        # Display the transformed image in the second column
        with col2:
            for key, value in transformed_img.items():
                st.image(value, caption=f"Transformed Image {key}")

    elif selected_transformation.lower() == "sobel":
        transformed_img = transformation(f"./input/{selected_image}")

        # Create two columns
        col1, col2 = st.columns(2)

        # Display the original image in the first column
        with col1:
            st.image(img, caption="Original Image")

        # Display the transformed image in the second column
        with col2:
            for key, value in transformed_img.items():
                st.image(value, caption=f"Transformed Image {key}")

    elif selected_transformation.lower() == "canny":
        transformed_img = transformation(f"./input/{selected_image}")

        # Create two columns
        col1, col2 = st.columns(2)

        # Display the original image in the first column
        with col1:
            st.image(img, caption="Original Image")

        # Display the transformed image in the second column
        with col2:
            st.image(transformed_img, caption="Transformed Image")

    else:
        if selected_transformation.lower() == "umbral":
            transformed_img = transformation(f"./input/{selected_image}", umbral_value)
        else:
            transformed_img = transformation(f"./input/{selected_image}")

        original_histogram = get_histogram(f"./input/{selected_image}")
        transformed_histogram = get_histogram_from_file(transformed_img)

        # Create two columns
        col1, col2 = st.columns(2)

        # Display the original image in the first column
        with col1:
            st.image(img, caption="Original Image")

        # Display the transformed image in the second column
        with col2:
            st.image(transformed_img, caption="Transformed Image")

        # * Histogram
        # Create a histogram with the values of the selected image
        col3, col4 = st.columns(2)

        # Display the histogram in the first column
        with col3:
            st.bar_chart(original_histogram)
            st.text("Original Histogram")

        # Display the histogram in the second column
        with col4:
            st.bar_chart(transformed_histogram)
            st.text("Transformed Histogram")

    # Add a separator
    st.markdown("---")

# * Image Gallery
# Create a gallery with all the images from the input folder
st.subheader("Image Gallery")
# Create a list to hold the columns
columns = []

# Create enough columns to hold all images
for i in range(0, len(images), 4):
    columns.extend(st.columns(4))

# Display each image in a column
for i, image in enumerate(images):
    with columns[i]:
        st.image(f"./input/{image}", caption=image)
