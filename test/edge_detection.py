import cv2
import numpy as np


def edge_detection(image_path, output_path):
    # Load the image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Check if image loading was successful
    if image is None:
        print("Could not open or find the image.")
        exit(0)

    # Apply the Sobel operator
    sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=7)
    sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=7)

    # Calculate the magnitude of the gradients
    magnitude = np.hypot(sobelx, sobely)

    # Normalize the magnitude to the range 0-255
    magnitude = magnitude / np.max(magnitude) * 255

    # Convert the magnitude to an 8-bit grayscale image
    magnitude = np.uint8(magnitude)

    # Save the result
    cv2.imwrite(output_path, magnitude)

    print(f"Edge detection completed. Result saved at {output_path}")


edge_detection("input/vatican_day.jpg", "output/sobel_edge_detection_7x7.bmp")
