# Image Transformation and Histogram

This repository contains a Python script that allows you to copy an image and apply various transformations to it. It also includes functions to calculate and plot the histogram of an image.

## Requirements

- Python 3.x
- PIL (Python Imaging Library)
- matplotlib
- opencv-python

## Usage

1. Clone the repository:

    ```bash
    git clone https://github.com/your-username/image-transformation-histogram.git
    ```

2. Install the required dependencies:

    ```bash
    pip install -r requirements.txt
    ```

3. Run the script:

    ```bash
    python main.py
    ```

## Available Transformations

- Inverse Transformation: Inverts the colors of the image.
- Umbral Transformation: Converts the image to black and white based on a threshold value.
- Grayscale Transformation: Converts the image to grayscale.
- Equalization: Enchances the image contrast using the equialization algorithm.
- Convolution Filters: Removes noise from the input image using various convolution filters, such as media and median filters.

## Output

The transformed images will be saved in the `output` directory. The histogram of the input image will be plotted and displayed.

## Image Processing with Convolutional Kernels

This module provides functionality to apply convolutional kernels to images, which is a fundamental operation in many image processing tasks. The module supports several types of kernels including Prewitt, Sobel, and Kirsch.

### Kernels Usage

The module provides the following functions:

- `converto_to_grayscale(image)`: Converts a color image to grayscale.
- `apply_filter(image, kernel)`: Applies a 3x3 kernel to an image.
- `apply_prewitt(image, pixel_by_pixel=False)`: Applies Prewitt operator to an image.
- `apply_sobel(image, pixel_by_pixel=False)`: Applies Sobel operator to an image.
- `apply_kirsch(image, pixel_by_pixel=False)`: Applies Kirsch operator to an image.

The `pixel_by_pixel` parameter in the `apply_*` functions determines whether the kernel is applied to each pixel individually (if `True`), or if the built-in `cv2.filter2D` function is used (if `False`).

The functions save the processed images to the `edge_detection_output` directory and return the processed images.

### Example

```python
import cv2
from kernels import converto_to_grayscale, apply_prewitt, apply_sobel, apply_kirsch

# Load the image
IMAGE_PATH = "input/domino.jpg"
original_image = cv2.imread(IMAGE_PATH)

# Convert the image to grayscale
image = converto_to_grayscale(original_image)

# Apply the kernels
prewitt_90, prewitt_0, prewitt_45, prewitt_minus_45 = apply_prewitt(image)
sobel_90, sobel_0 = apply_sobel(image)
kirsch_90, kirsch_0, kirsch_45, kirsch_minus_45 = apply_kirsch(image)

print("Done!")
```

This will apply the Prewitt, Sobel, and Kirsch operators to the image and save the results. The processed images can be found in the `edge_detection_output` directory.

## Fourier Transform and Frequency Domain Filtering

This Python script demonstrates the application of Fourier Transform and Frequency Domain Filtering on an image. It uses the `numpy` and `cv2` libraries to perform these operations.

### How it works

1. The script first reads an image file named `figuras.bmp` in grayscale mode.
2. It then applies the Fourier Transform to the image and shifts the zero-frequency component to the center of the spectrum.
3. The script displays the original image and the logarithm of the absolute value of the Fourier Transform.
4. It then creates an ideal low-pass filter and an ideal high-pass filter in the frequency domain.
5. The script applies these filters to the Fourier Transform of the image, effectively removing high-frequency (for the low-pass filter) or low-frequency (for the high-pass filter) components.
6. The script displays the magnitude spectrum of the filtered Fourier Transforms.
7. Finally, it applies the inverse Fourier Transform to the filtered images, converts them back to the spatial domain, and displays the filtered images.

## Fourier Filters Usage

Run the script with the command `python fourier.py`. The script will display the original image, the Fourier Transform, the filters, and the filtered images in separate windows. Press any key to close the windows and end the script.

## Licence

[MIT](https://choosealicense.com/licenses/mit/)
