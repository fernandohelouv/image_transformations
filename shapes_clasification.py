""" This script is used to classify shapes using the Fourier Descriptors """

import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read image
img = cv2.imread("shapes/square45.png", 2)

# Calculate the area of the figure on the image
# The area is the number of pixels that are black
area = np.sum(img == 0)
print("Area:", area)

# Calculate the center of the figure
# The center is the average of the coordinates of the pixels that are black
x, y = np.where(img == 0)
center = (np.mean(x), np.mean(y))
print("Center:", center)

# Calculate the center of mass of the figure
# The center of mass is the sum of the coordinates of the pixels that are black
# divided by the area
x, y = np.where(img == 0)
center_mass = (np.sum(x) / area, np.sum(y) / area)
print("Center of Mass:", center_mass)

# Calculate the fourier polar desciptors

# Calculate the distance from the center of mass to
# the edge of the figure. Each distance would be taken as a radius separated by N degrees
# N is calculated as 360 / number of points

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

# Apply the DFT to the distances array
# dft_r = np.fft.fft(r)

# Apply the DFT by "hand"
dft_r = np.zeros(N, dtype=complex)

for m in range(N):
    for n in range(N):
        dft_r[m] += r[n] * np.exp(-2j * np.pi * n * m / N)

    # Calculate the magnitude of the Fourier Transform
    dft_r[m] = np.sqrt(dft_r[m].real ** 2 + dft_r[m].imag ** 2)

# Shift the image to the center
dft_r = np.fft.fftshift(dft_r)

# Plot the absolute values of the DFT coefficients side by side with the distances, and the original image
plt.subplot(1, 3, 1)
plt.plot(r)
plt.title("Characteristics Vector")
plt.xlabel("n")
plt.ylabel("r(n)")

plt.subplot(1, 3, 2)
plt.plot(dft_r)
plt.title("DFT of Characteristics Vector")
plt.xlabel("n")
plt.ylabel("|F(n)|")

plt.subplot(1, 3, 3)
plt.imshow(img, cmap="gray")
plt.title("Original Image")

plt.show()
