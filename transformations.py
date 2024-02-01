""" This script copies an image and applies a transformation to it. """

import cv2
import numpy as np
from PIL import Image


def get_histogram(image) -> list:
    # Get the image's width and height
    width, height = image.size

    # Create a list of 256 elements
    histogram = [0] * 256

    for x in range(width):
        for y in range(height):
            # Get the pixel at the current position
            pixel = image.getpixel((x, y))

            if isinstance(pixel, int):
                luminance = pixel
            else:
                # Get the luminance of the pixel
                luminance = int(0.299 * pixel[0] + 0.587 * pixel[1] + 0.114 * pixel[2])

            # Increment the value in the histogram
            histogram[luminance] += 1

    return histogram


def grayscale(image) -> Image:
    # Get the image's width and height
    x, y = image.size

    # Create a new image with the same size
    new_image = Image.new("RGB", (x, y))

    for i in range(x):
        for j in range(y):
            # Get the pixel at the current position
            pixel = image.getpixel((i, j))

            # Get the luminance of the pixel
            luminance = int(0.299 * pixel[0] + 0.587 * pixel[1] + 0.114 * pixel[2])

            # Set the new pixel
            new_image.putpixel((i, j), (luminance, luminance, luminance))

    return new_image


def invert(image) -> Image:
    # Get the image's width and height
    width, height = image.size

    # Create a new image with the same size
    new_image = Image.new("RGB", (width, height))

    for x in range(width):
        for y in range(height):
            # Get the pixel at the current position
            pixel = image.getpixel((x, y))

            if isinstance(pixel, int):
                transformed_pixel = 255 - pixel
            else:
                transformed_pixel = tuple(255 - p for p in pixel)

            new_image.putpixel((x, y), transformed_pixel)

    return new_image


def umbral(image, threshold: int = 127) -> Image:
    # Turn the image into grayscale
    image = grayscale(image)

    # Get the image's width and height
    width, height = image.size

    # Create a new image with the same size
    new_image = Image.new("RGB", (width, height))

    for x in range(width):
        for y in range(height):
            # Get the pixel at the current position
            pixel = image.getpixel((x, y))
            transformed_pixel = tuple(0 if p < threshold else 255 for p in pixel)
            new_image.putpixel((x, y), transformed_pixel)

    return new_image


def enhance_contrast(image) -> Image:
    # Get the image's width and height
    width, height = image.size

    # Create a new image with the same size
    new_image = Image.new("RGB", (width, height))

    # Get the histogram of the image
    histogram = get_histogram(image)

    # Get the minimum and maximum luminance
    min_luminance = histogram.index(next(filter(lambda x: x > 0, histogram)))
    print("min luminance", min_luminance)
    max_luminance = histogram.index(next(filter(lambda x: x < 256, histogram[::-1])))
    print("max luminance", max_luminance)

    for x in range(width):
        for y in range(height):
            # Get the pixel at the current position
            pixel = image.getpixel((x, y))

            # Get the luminance of the pixel
            luminance = int(0.299 * pixel[0] + 0.587 * pixel[1] + 0.114 * pixel[2])

            # Apply the transformation
            transformed_pixel = tuple(
                int(
                    (luminance - min_luminance)
                    * (255 / (max_luminance - min_luminance))
                )
                for luminance in pixel
            )

            # Set the new pixel
            new_image.putpixel((x, y), transformed_pixel)

    return new_image


def equalization(image) -> Image:
    """Equalize the histogram of an image.

    Args:
        image_url (str, optional): Input image url. Defaults to "./input/image.jpg".
    """

    # Get the histogram of the image
    histogram = get_histogram(image)
    cumulative_sum = 0
    scale_factor = 255 / sum(histogram)
    new_histogram = []

    # Calculate the cumulative distribution function (CDF)
    for luminance in histogram:
        cumulative_sum += luminance
        new_value = round(cumulative_sum * scale_factor)
        new_histogram.append(new_value)

    # Apply the transformation
    width, height = image.size
    new_image = Image.new("RGB", (width, height))

    for x in range(width):
        for y in range(height):
            pixel = image.getpixel((x, y))
            if isinstance(pixel, int):
                new_pixel = new_histogram[pixel]
                new_image.putpixel((x, y), (new_pixel, new_pixel, new_pixel))
            else:
                new_pixel = tuple(new_histogram[val] for val in pixel)
                new_image.putpixel((x, y), new_pixel)

    return new_image


def media_filter(image) -> Image:
    """Apply the media filter to an image.

    Args:
        image: previously loaded image
    """

    image = image.convert("L")  # Convert to grayscale

    # Get image size
    width, height = image.size

    # Create a new image in grayscale
    new_image = Image.new("L", (width, height))

    # * Apply convolution filter
    # ignore the edge pixels for simplicity (1 to width-1 instead of 0 to width)
    for x in range(1, width - 1):
        # ignore edge pixels for simplicity (1 to height-1 instead of 0 to height)
        for y in range(1, height - 1):
            # Get pixel neighborhood
            neighbors = []

            for i in range(-1, 2):
                for j in range(-1, 2):
                    neighbors.append(image.getpixel((x + i, y + j)))

            # TODO: Check if the pixel is a tuple or an int
            if isinstance(neighbors[0], int):
                # Calculate the media (average)
                average = sum(neighbors) // 9

                # Save the new pixel in the new image
                new_image.putpixel((x, y), average)

    return new_image


def median_filter(image) -> Image:
    """Apply the median filter to an image.

    Args:
        image: previously loaded image
    """

    # Original image
    image = image.convert("L")  # Convert to grayscale

    # Get image size
    width, height = image.size

    # Create a new image
    new_image = Image.new("L", (width, height))  # Grayscale image

    # Apply the median filter
    for x in range(1, width - 1):
        for y in range(1, height - 1):
            # Get the 3x3 neighborhood
            neighbors = []
            for i in range(-1, 2):
                for j in range(-1, 2):
                    neighbors.append(image.getpixel((x + i, y + j)))

            # Sort the neighborhood
            neighbors.sort()

            # Get the median
            median = neighbors[4]

            # Save the median pixel
            new_image.putpixel((x, y), median)

    return new_image


def apply_filter(image, kernel) -> dict:
    """This function applies a kernel to an image, pixel by pixel.
    It only works with 3x3 kernels.

    Args:
        image (_type_): image to apply the kernel to
        kernel (_type_): kernel to apply to the image

    Returns:
        _type_: _description_
    """

    # Create a new image to store the result
    result = np.zeros(image.shape, dtype=np.uint8)

    # Apply the kernels to the image pixel by pixel
    for i in range(1, image.shape[0] - 1):
        for j in range(1, image.shape[1] - 1):
            result[i, j] = np.sum(image[i - 1 : i + 2, j - 1 : j + 2] * kernel)

    return result


def apply_prewitt(image, pixel_by_pixel=False) -> dict:

    # Convert the image from PIL to numpy
    image = np.array(image)

    gain = 0

    # Define the kernels
    kernel_90 = np.array(
        [[1, gain, -1], [1, gain, -1], [1, gain, -1]], dtype=np.float32
    )

    kernel_0 = np.array([[1, 1, 1], [gain, gain, gain], [-1, -1, -1]], dtype=np.float32)

    kernel_45 = np.array(
        [[1, 1, gain], [1, gain, -1], [gain, -1, -1]], dtype=np.float32
    )

    kernel_minus_45 = np.array(
        [[gain, 1, 1], [-1, gain, 1], [-1, -1, gain]], dtype=np.float32
    )

    results = {}

    if pixel_by_pixel:
        # Apply the kernels to the image
        result_90 = apply_filter(image, kernel_90)
        result_0 = apply_filter(image, kernel_0)
        result_45 = apply_filter(image, kernel_45)
        result_minus_45 = apply_filter(image, kernel_minus_45)

    else:
        # Apply the kernels to the image
        result_90 = cv2.filter2D(image, -1, kernel_90)
        result_0 = cv2.filter2D(image, -1, kernel_0)
        result_45 = cv2.filter2D(image, -1, kernel_45)
        result_minus_45 = cv2.filter2D(image, -1, kernel_minus_45)

    results["90"] = result_90
    results["0"] = result_0
    results["45"] = result_45
    results["-45"] = result_minus_45

    # Combine the results
    result = np.zeros(result_90.shape, dtype=np.uint8)
    result = cv2.add(result, result_90)
    result = cv2.add(result, result_0)
    result = cv2.add(result, result_45)
    result = cv2.add(result, result_minus_45)

    dict_result = {
        "90": result_90,
        "0": result_0,
        "45": result_45,
        "-45": result_minus_45,
        "combined": result,
    }

    return dict_result


def apply_sobel(image, pixel_by_pixel=False) -> dict:
    # Convert the image from PIL to numpy
    image = np.array(image)

    gain = 0
    # Define the kernels
    kernel_90 = np.array(
        [[-1, gain, 1], [-2, gain, 2], [-1, gain, 1]], dtype=np.float32
    )

    kernel_0 = np.array([[-1, -2, -1], [gain, gain, gain], [1, 2, 1]], dtype=np.float32)

    if pixel_by_pixel:
        # Apply the kernels to the image
        result_90 = apply_filter(image, kernel_90)
        result_0 = apply_filter(image, kernel_0)
    else:
        # Apply the kernels to the image
        result_90 = cv2.filter2D(image, -1, kernel_90)
        result_0 = cv2.filter2D(image, -1, kernel_0)

    # Combine the results
    result = np.zeros(result_90.shape, dtype=np.uint8)
    result = cv2.add(result, result_90)
    result = cv2.add(result, result_0)

    dict_result = {"90": result_90, "0": result_0, "combined": result}

    return dict_result


def apply_kirsch(image, pixel_by_pixel=False) -> dict:
    # Convert the image from PIL to numpy
    image = np.array(image)

    gain = 2

    # Define the kernels
    kernel_0 = np.array(
        [[-1, -1, -1], [gain, gain, gain], [-1, -1, -1]], dtype=np.float32
    )

    kernel_45 = np.array(
        [[-1, -1, gain], [-1, gain, -1], [gain, -1, -1]], dtype=np.float32
    )

    kernel_90 = np.array(
        [[-1, gain, -1], [-1, gain, -1], [-1, gain, -1]], dtype=np.float32
    )

    kernel_minus_45 = np.array(
        [[gain, -1, -1], [-1, gain, -1], [-1, -1, gain]], dtype=np.float32
    )

    if pixel_by_pixel:
        # Apply the kernels to the image
        result_90 = apply_filter(image, kernel_90)
        result_0 = apply_filter(image, kernel_0)
        result_45 = apply_filter(image, kernel_45)
        result_minus_45 = apply_filter(image, kernel_minus_45)
    else:
        # Apply the kernels to the image
        result_90 = cv2.filter2D(image, -1, kernel_90)
        result_0 = cv2.filter2D(image, -1, kernel_0)
        result_45 = cv2.filter2D(image, -1, kernel_45)
        result_minus_45 = cv2.filter2D(image, -1, kernel_minus_45)

    # Combine the results
    result = np.zeros(result_90.shape, dtype=np.uint8)
    result = cv2.add(result, result_90)
    result = cv2.add(result, result_0)
    result = cv2.add(result, result_45)
    result = cv2.add(result, result_minus_45)

    dict_result = {
        "90": result_90,
        "0": result_0,
        "45": result_45,
        "-45": result_minus_45,
        "combined": result,
    }

    return dict_result


def cany(image) -> dict:
    # Convert the image from PIL to numpy
    image = np.array(image)

    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    canny_output = cv2.Canny(image, 80, 150)

    dict_result = {"canny": canny_output}

    return dict_result
