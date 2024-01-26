"""This script uses the get_clasify_data function and the previously calculated data to indentify the shape on an image"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pickle
from shapes_training import get_clasify_data

# Read all the images in the shapes_to_clasify folder
main_path = "shapes_to_clasify"
images = os.listdir(main_path)

# Get the data for each image
for image in images:
    r, dft_r = get_clasify_data(os.path.join(main_path, image))

    # Load the data for the shapes with pickle
    # all files are stored in the shapes_data folder
    shapes_data_path = "shapes_data"
    data = {}
    for shape in os.listdir(shapes_data_path):
        with open(os.path.join(shapes_data_path, shape), "rb") as file:
            data[shape] = pickle.load(file)

    # Calculate the distance between the shape and the image
    distances = {}
    for shape in data:
        # Calculate the distance between the shape and the image
        distances[shape] = np.linalg.norm(data[shape]["dft_r"] - dft_r)

    # Get the shape with the minimum distance
    shape = min(distances, key=distances.get)
    print(f"Shape: {shape}")
    print(f"Distance: {distances[shape]}")
