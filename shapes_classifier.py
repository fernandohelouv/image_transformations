"""This script uses the get_clasify_data function and the previously calculated data to indentify the shape on an image"""

import os
import cv2
import matplotlib.pyplot as plt
import pickle
from shapes_classifier_training import get_clasify_data

# Load the data for the shapes with pickle
# all files are stored in the shapes_data folder
shapes_data_path = "shapes_data"
data = {}
for shape in os.listdir(shapes_data_path):
    with open(os.path.join(shapes_data_path, shape), "rb") as file:
        data[shape] = pickle.load(file)

# print(data.keys())

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
        print(f"Image: {image_base} -> {shape_base} ✅ ")
    else:
        print(f"Image: {image_base} -> {shape_base} ❌")

    # Store the results
    results[image] = {
        "p_shape": shape_base,
        "distance": distances[shape],
        "r": r,
        "dft_r": dft_r,
    }

exit()

# Plot the results of each image
for image in results:
    plt.rcParams["figure.figsize"] = [15, 6]
    img = cv2.imread(os.path.join(main_path, image), 2)

    plt.subplot(1, 3, 1)
    plt.plot(results[image]["r"])
    plt.title(f"Characteristics Vector for {image}")
    plt.xlabel("n")
    plt.ylabel("r(n)")

    plt.subplot(1, 3, 2)
    plt.plot(results[image]["dft_r"])
    plt.title(f"DFT of Characteristics Vector for {image}")
    plt.xlabel("n")
    plt.ylabel("|F(n)|")

    plt.subplot(1, 3, 3)
    plt.imshow(img, cmap="gray")
    plt.title(f"Original: {image} -> Prediction: {results[image]['p_shape']}")

    plt.show()
