""" Fourier Transform and Frequency Domain Filtering """

import numpy as np
import cv2

img = cv2.imread("input/figuras.bmp", 2)
x, y = img.shape[0:2]
print("Image Shape:", x, y)

frr = np.fft.fft2(img)
frr = np.fft.fftshift(frr)

frr_abs = np.abs(frr)

frr_log = np.log10(frr_abs)

cv2.imshow("Original", img)

img_frr = np.uint8(255 * frr_log / np.max(frr_log))
cv2.imshow("Fourier Log", img_frr)

# Ideal Low Pass Filter
ideal_low_pass = np.zeros((x, y))

for i in range(x):
    for j in range(y):
        d = np.sqrt((i - x / 2) ** 2 + (j - y / 2) ** 2)
        if d <= 30:
            ideal_low_pass[i, j] = 1
        else:
            ideal_low_pass[i, j] = 0

cv2.imshow("Ideal Low Pass Filter", ideal_low_pass)

# Ideal High Pass Filter
ideal_high_pass = 1 - ideal_low_pass

cv2.imshow("Ideal High Pass Filter", np.uint8(255 * ideal_high_pass))

# Frequency Domain Filtering
Guv_low_pass = ideal_low_pass * frr
Guv_high_pass = ideal_high_pass * frr

# Magnitude Spectrum
ideal_low_pass_filter_abs = np.abs(Guv_low_pass)
ideal_low_pass_filter_abs = np.uint8(
    255
    * np.log10(ideal_low_pass_filter_abs)
    / np.max(np.log10(ideal_low_pass_filter_abs))
)

cv2.imshow("Frequency Domain Filtering (Low Pass)", ideal_low_pass_filter_abs)

# Inverse Fourier Transform
gxy_low_pass = np.fft.ifft2(Guv_low_pass)
gxy_low_pass = np.abs(gxy_low_pass)
gxy_low_pass = np.uint8(255 * gxy_low_pass / np.max(gxy_low_pass))

# Show filtered image (Low Pass)
cv2.imshow("Filtered Image (Low Pass)", gxy_low_pass)

# Magnitude Spectrum for High Pass Filter
ideal_high_pass_filter_abs = np.abs(Guv_high_pass)
ideal_high_pass_filter_abs = np.uint8(
    255
    * np.log10(ideal_high_pass_filter_abs)
    / np.max(np.log10(ideal_high_pass_filter_abs))
)

cv2.imshow("Frequency Domain Filtering (High Pass)", ideal_high_pass_filter_abs)

# Inverse Fourier Transform for High Pass Filter
gxy_high_pass = np.fft.ifft2(Guv_high_pass)
gxy_high_pass = np.abs(gxy_high_pass)
gxy_high_pass = np.uint8(255 * gxy_high_pass / np.max(gxy_high_pass))

# Show filtered image (High Pass)
cv2.imshow("Filtered Image (High Pass)", gxy_high_pass)

cv2.waitKey(0)
cv2.destroyAllWindows()
