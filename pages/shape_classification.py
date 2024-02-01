import os
from unittest import result
import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt

# import from ../shapes_classifier.py
from shapes_classifier import classify

# Get the list of images
images = os.listdir("./shapes")

# * Image Gallery
# Create a gallery with all the images from the input folder
st.subheader("Available Shapes to Classify")
# Create a list to hold the columns
columns = []

# Create enough columns to hold all images
for i in range(0, len(images), 4):
    columns.extend(st.columns(4))

# Display each image in a column
for i, image in enumerate(images):
    with columns[i]:
        st.image(f"./shapes/{image}", caption=image)

# Add a separator
st.markdown("---")

# * Shape Classification
if st.button("Classify Shapes"):
    results = classify()

    # Display the results
    for image in results:
        # Extract the base names of the image and the shape
        shape_base = results[image]["p_shape"]
        image_base = image.replace(".png", "")

        st.header(f"Results for {image}")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("##")
            st.markdown("##")
            st.subheader(f"Predicted Shape: {results[image]['status']}")
            st.subheader(f"{results[image]['p_shape']}")
        with col2:
            st.image(
                f"./shapes_to_classify/{results[image]['image']}",
                caption=results[image]["p_shape"],
            )

        col3, col4 = st.columns(2)

        with col3:
            st.write(f"Characteristics Vector")
            st.line_chart(results[image]["r"])

        with col4:
            st.write(f"DFT of Characteristics Vector")
            fig, ax = plt.subplots()

            # Plot the real part
            ax.plot(results[image]["dft_r"].real, label="Real")

            # Plot the imaginary part
            # ax.plot(results[image]["dft_r"].imag, label="Imaginary")

            # Set labels and legend
            ax.set_xlabel("n")
            ax.set_ylabel("DFT Value")
            ax.legend()

            st.pyplot(fig)

        st.markdown("---")
