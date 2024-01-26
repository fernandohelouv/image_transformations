# Image Transformation and Histogram

This repository contains a Python script that allows you to copy an image and apply various transformations to it. It also includes functions to calculate and plot the histogram of an image.

## Requirements

- Python 3.x
- PIL (Python Imaging Library)
- matplotlib
- opencv-python
- pickle

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

## Shape Classifier Training Script

This Python script is used to classify shapes using Fourier Descriptors. It reads images of shapes, calculates the Fourier Descriptors for each shape, and saves the results for later use in shape classification.

### How Shape Classifier Training Script Works

The script works by performing the following steps for each image:

1. Reads the image and calculates the center of mass of the shape in the image.

2. Calculates a set of distances from the center of mass to the edge of the shape at different angles.

3. Normalizes the distances to be between 0 and 1.

4. Applies the Discrete Fourier Transform (DFT) to the set of distances to get the Fourier Descriptors.

5. Shifts the Fourier Descriptors to center them.

6. If in training mode, saves the normalized distances and Fourier Descriptors to a file.

### Shape Classifier Training Script Usage

To use this script, you need to have a folder of images of shapes that you want to classify. The images should be in PNG format and the shapes should be black on a white background.

By default, the script reads images from a folder named "shapes".

The script saves the classification data for each image in a separate file in the "shapes_data" folder. The data for each image is saved in a Python pickle file with the same name as the image file, but with a ".pkl" extension.

## Shape Classifier Script

This Python script uses the data calculated by the shapes_classifier_training.py script to identify the shape in an image.

### How Shape Classifier Script Works

The script works by performing the following steps:

1. Loads the previously calculated data for each shape from the "shapes_data" folder.
2. Reads the images in the "shapes_to_classify" folder.
3. For each image, it calculates the characteristic vector and its Discrete Fourier Transform (DFT) using the get_clasify_data function from the shapes_classifier_training module.
4. It then calculates the Euclidean distance between the DFT of the image and the DFT of each shape.
5. The shape with the smallest distance is considered the predicted shape for the image.
6. The script then prints whether the predicted shape matches the actual shape (based on the image file name).
7. Finally (optional), it plots the characteristic vector, its DFT, and the original image for each image.

### Shape Classifier Script Usage

To use this script, you need to have a folder of images of shapes that you want to classify. The images should be in PNG format and the shapes should be black on a white background.

You also need to have previously run the shapes_classifier_training.py script to calculate the data for each shape.

By default, the script reads images from a folder named "shapes_to_classify".

## Licence

[MIT](https://choosealicense.com/licenses/mit/)
