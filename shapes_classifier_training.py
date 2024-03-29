""" This script is used to classify shapes using the Fourier Descriptors """

import os
import cv2
from matplotlib import pyplot as plt
import numpy as np
import pickle


def get_clasify_data(img_path: str, training: bool = False):
    """
    This function is used to classify shapes using the Fourier Descriptors

    Args:
        img_path (str, optional): Path to the image to analyze. Defaults to "shapes/square45.png".
    """

    # Read image
    img = cv2.imread(img_path, 2)

    # Calculate the center of the figure
    # The center is the average of the coordinates of the pixels that are black
    x, y = np.where(img == 0)
    center_mass = (np.mean(x), np.mean(y))

    N = 50
    r = np.zeros(N)
    for i in range(0, N - 1):
        # Calculate the angle
        angle = i * 2 * np.pi / N
        # print("Rad:", angle)
        # print("Deg:", np.rad2deg(angle))

        # Calculate the line from the center of mass in the direction of the angle
        x_line = center_mass[0] + np.cos(angle) * np.arange(img.shape[0])
        y_line = center_mass[1] + np.sin(angle) * np.arange(img.shape[1])

        # Iterate over the line coordinates
        for x, y in zip(x_line, y_line):
            # Check if the point is a black pixel in the image
            if img[int(x), int(y)] == 0:
                # Continue to the next point
                continue

            # If the point is not a black pixel, it means we found the edge of the square
            # Calculate the distance from the center of mass to the point
            distance = np.sqrt((x - center_mass[0]) ** 2 + (y - center_mass[1]) ** 2)

            # Break the loop as we found the edge for this angle
            break

        # Store the distance
        r[i] = distance

    # Normalize the distances to be between 0 and 1
    r = r / r.max()

    # Apply the DFT by "hand"
    dft_r = np.zeros(N, dtype=complex)

    for m in range(N):
        for n in range(N):
            dft_r[m] += r[n] * np.exp(-2j * np.pi * n * m / N)

        # Calculate the magnitude of the Fourier Transform
        dft_r[m] = np.sqrt(dft_r[m].real ** 2 + dft_r[m].imag ** 2)

    # Shift the image to the center
    dft_r = np.fft.fftshift(dft_r)

    if training:
        # Save the results to a file
        save_path = img_path.replace(".png", ".pkl")
        save_path = save_path.replace("shapes", "shapes_data")
        with open(save_path, "wb") as f:
            data = {"r": r, "dft_r": dft_r}
            pickle.dump(data, f)

    # return the results
    return r, dft_r


# run function if called directly
if __name__ == "__main__":
    # Read all the images in the shapes folder
    main_path = "shapes"
    images = os.listdir(main_path)

    clasification_data = {}
    # Get the data for each image
    for image in images:
        # Get the data for the image
        r, dft_r = get_clasify_data(os.path.join(main_path, image), training=True)

        # plot the r
        plt.figure()
        plt.plot(r)
        plt.title(f"r for {image}")

    # Show the plots
    plt.show()
