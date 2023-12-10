# Image Transformation and Histogram

This repository contains a Python script that allows you to copy an image and apply various transformations to it. It also includes functions to calculate and plot the histogram of an image.

## Requirements

- Python 3.x
- PIL (Python Imaging Library)
- matplotlib

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

## Licence

[MIT](https://choosealicense.com/licenses/mit/)
