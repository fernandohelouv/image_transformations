"""This script uses the get_clasify_data function and the previously calculated data to indentify the shape on an image"""

import os
import pickle
from shapes_classifier_training import get_clasify_data


def classify() -> dict:
    # Load the data for the shapes with pickle
    # all files are stored in the shapes_data folder
    shapes_data_path = "shapes_data"
    data = {}
    for shape in os.listdir(shapes_data_path):
        with open(os.path.join(shapes_data_path, shape), "rb") as file:
            data[shape] = pickle.load(file)

    # Read all the images in the shapes_to_classify folder
    main_path = "shapes_to_classify"
    images = os.listdir(main_path)

    results = {}

    for image in images:
        # Get the data for each image
        r, dft_r = get_clasify_data(os.path.join(main_path, image))

        distances = {}
        # Calculate the distance between the shape and the image
        # comparing the DFT of the shape with the DFT of the image
        for shape in data:
            shape_vector = data[shape]["dft_r"]

            # Initialize a variable to hold the sum of squares
            sum_of_squares = 0

            # Loop over the elements in the two vectors
            for i in range(len(shape_vector)):
                # Get the difference between the two elements
                diff = shape_vector[i] - dft_r[i]

                # Square the difference and add it to the sum of squares
                sum_of_squares += diff**2

            # Calculate the Euclidean distance by taking the square root of the sum of squares
            euclidean_distance = sum_of_squares**0.5

            # Store the distance
            distances[shape] = euclidean_distance

        # Get the shape with the minimum distance
        shape = min(distances, key=distances.get)

        # Extract the base names of the image and the shape
        shape_base = shape.replace(".pkl", "")
        image_base = image.replace(".png", "")

        # Check if the base names are the same
        if shape_base in image_base:
            status = "✅"
            print(f"Image: {image_base} -> {shape_base} ✅ ")
        else:
            status = "❌"
            print(f"Image: {image_base} -> {shape_base} ❌")

        # Store the results
        results[image] = {
            "image": image,
            "status": status,
            "p_shape": shape_base,
            "distance": distances[shape],
            "r": r,
            "dft_r": dft_r,
        }

    return results
