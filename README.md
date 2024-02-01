# Image Transformations

This repository contains an Streamlit App that allows you to apply various transformations to an image.

![GUI Screenshot](<Screenshot 2024-01-31 at 20.58.02.jpg>)

## Available Transformations

* Enhance Contrast
* Turn to Gray Scale
* Umbral Binarization
* Invert Color
* Luminance Equalization
* Noise Filter (Media)
* Noise Filter (Median)
* Edge Detection (Prewitt)
* Edge Detection (Kirsch)
* Edge Detection (Sobel)
* Edge Detection (Canny)

## Usage

1. Clone the repository:

    ```bash
    git clone https://github.com/your-username/image-transformation-histogram.git
    ```

2. Install the required dependencies:

    ```bash
    pip install -r requirements.txt
    ```

3. Run the app:

    ```bash
    streamlit run app.py
    ```

## Not implemente din the Streamlit app (yet):

### Fourier Transform and Frequency Domain Filtering

This Python script demonstrates the application of Fourier Transform and Frequency Domain Filtering on an image. It uses the `numpy` and `cv2` libraries to perform these operations.

#### How it works

1. The script first reads an image file named `figuras.bmp` in grayscale mode.
2. It then applies the Fourier Transform to the image and shifts the zero-frequency component to the center of the spectrum.
3. The script displays the original image and the logarithm of the absolute value of the Fourier Transform.
4. It then creates an ideal low-pass filter and an ideal high-pass filter in the frequency domain.
5. The script applies these filters to the Fourier Transform of the image, effectively removing high-frequency (for the low-pass filter) or low-frequency (for the high-pass filter) components.
6. The script displays the magnitude spectrum of the filtered Fourier Transforms.
7. Finally, it applies the inverse Fourier Transform to the filtered images, converts them back to the spatial domain, and displays the filtered images.

### Fourier Filters Usage

Run the script with the command `python fourier.py`. The script will display the original image, the Fourier Transform, the filters, and the filtered images in separate windows. Press any key to close the windows and end the script.

### Shape Classifier Training Script

This Python script is used to classify shapes using Fourier Descriptors. It reads images of shapes, calculates the Fourier Descriptors for each shape, and saves the results for later use in shape classification.

#### How Shape Classifier Training Script Works

The script works by performing the following steps for each image:

1. Reads the image and calculates the center of mass of the shape in the image.

2. Calculates a set of distances from the center of mass to the edge of the shape at different angles.

3. Normalizes the distances to be between 0 and 1.

4. Applies the Discrete Fourier Transform (DFT) to the set of distances to get the Fourier Descriptors.

5. Shifts the Fourier Descriptors to center them.

6. If in training mode, saves the normalized distances and Fourier Descriptors to a file.

#### Shape Classifier Training Script Usage

To use this script, you need to have a folder of images of shapes that you want to classify. The images should be in PNG format and the shapes should be black on a white background.

By default, the script reads images from a folder named "shapes".

The script saves the classification data for each image in a separate file in the "shapes_data" folder. The data for each image is saved in a Python pickle file with the same name as the image file, but with a ".pkl" extension.

### Shape Classifier Script

This Python script uses the data calculated by the shapes_classifier_training.py script to identify the shape in an image.

#### How Shape Classifier Script Works

The script works by performing the following steps:

1. Loads the previously calculated data for each shape from the "shapes_data" folder.
2. Reads the images in the "shapes_to_classify" folder.
3. For each image, it calculates the characteristic vector and its Discrete Fourier Transform (DFT) using the get_clasify_data function from the shapes_classifier_training module.
4. It then calculates the Euclidean distance between the DFT of the image and the DFT of each shape.
5. The shape with the smallest distance is considered the predicted shape for the image.
6. The script then prints whether the predicted shape matches the actual shape (based on the image file name).
7. Finally (optional), it plots the characteristic vector, its DFT, and the original image for each image.

#### Shape Classifier Script Usage

To use this script, you need to have a folder of images of shapes that you want to classify. The images should be in PNG format and the shapes should be black on a white background.

You also need to have previously run the shapes_classifier_training.py script to calculate the data for each shape.

By default, the script reads images from a folder named "shapes_to_classify".

## Licence

[MIT](https://choosealicense.com/licenses/mit/)
