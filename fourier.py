""" Fourier Transform and Frequency Domain Filtering """

import numpy as np
import cv2

img = cv2.imread("input/figuras.bmp", 2)
x, y = img.shape[0:2]
print("Image Shape:", x, y)

# Fourier Transform
frr = np.fft.fft2(img)
# Shift the image to the center
frr = np.fft.fftshift(frr)

# Remove the imaginary part
frr_abs = np.abs(frr)

# Logarithmic Transformation to visualize the spectrum
# large change on large values, small change on small values
frr_log = np.log10(frr_abs)

cv2.imshow("Original", img)

# Scale the values of the array to 0-255
img_frr = np.uint8(255 * frr_log / np.max(frr_log))
cv2.imshow("Fourier Log", img_frr)

# Ideal Low Pass Filter
# Create a matrix of zeros with the same shape as the image
ideal_low_pass = np.zeros((x, y))

# Create the filter
for i in range(x):
    for j in range(y):
        # Calculate the distance from the center of the image to the current pixel
        d = np.sqrt((i - x / 2) ** 2 + (j - y / 2) ** 2)
        if d <= 30:
            # If the distance is less than 30, the pixel is white
            ideal_low_pass[i, j] = 1
        else:
            # If the distance is greater than 30, the pixel is black
            ideal_low_pass[i, j] = 0

cv2.imshow("Ideal Low Pass Filter", ideal_low_pass)

# Ideal High Pass Filter
# Convert the low pass filter to high pass filter
# if the pixel is white, it becomes black
# if the pixel is black, it becomes white
ideal_high_pass = 1 - ideal_low_pass

# Show the filter, white pixels are the ones that will be filtered
# black pixels are the ones that will be kept
cv2.imshow("Ideal High Pass Filter", np.uint8(255 * ideal_high_pass))

# Frequency Domain Filtering
# Multiply the Fourier Transform by the filter
Guv_low_pass = ideal_low_pass * frr
Guv_high_pass = ideal_high_pass * frr

# Magnitude Spectrum
# Convert the complex numbers to absolute values
ideal_low_pass_filter_abs = np.abs(Guv_low_pass)

# Scale the values of the array to 0-255
ideal_low_pass_filter_abs = np.uint8(
    255
    * np.log10(ideal_low_pass_filter_abs)
    / np.max(np.log10(ideal_low_pass_filter_abs))
)

cv2.imshow("Frequency Domain Filtering (Low Pass)", ideal_low_pass_filter_abs)

# Inverse Fourier Transform
# Convert the filtered image to the spatial domain
# by applying the inverse Fourier Transform
gxy_low_pass = np.fft.ifft2(Guv_low_pass)

# Convert the complex numbers to absolute values
gxy_low_pass = np.abs(gxy_low_pass)

# Scale the values of the array to 0-255
gxy_low_pass = np.uint8(255 * gxy_low_pass / np.max(gxy_low_pass))

# Show filtered image (Low Pass)
cv2.imshow("Filtered Image (Low Pass)", gxy_low_pass)

# Magnitude Spectrum for High Pass Filter
ideal_high_pass_filter_abs = np.abs(Guv_high_pass)

# Scale the values of the array to 0-255
ideal_high_pass_filter_abs = np.uint8(
    255
    * np.log10(ideal_high_pass_filter_abs)
    / np.max(np.log10(ideal_high_pass_filter_abs))
)

cv2.imshow("Frequency Domain Filtering (High Pass)", ideal_high_pass_filter_abs)

# Inverse Fourier Transform for High Pass Filter
gxy_high_pass = np.fft.ifft2(Guv_high_pass)

# Convert the complex numbers to absolute values
gxy_high_pass = np.abs(gxy_high_pass)

# Scale the values of the array to 0-255
gxy_high_pass = np.uint8(255 * gxy_high_pass / np.max(gxy_high_pass))

# Show filtered image (High Pass)
cv2.imshow("Filtered Image (High Pass)", gxy_high_pass)

cv2.waitKey(0)
cv2.destroyAllWindows()
