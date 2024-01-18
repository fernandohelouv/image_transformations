""" This script copies an image and applies a transformation to it. """

from PIL import Image
import matplotlib.pyplot as plt


def inverse_transformation(pixel: list) -> list:
    """Apply inverse color tranformation to a given pixel.

    Args:
        pixel (list): RGB values for a given pixel.

    Returns:
        list: New RGB inversed pixel values.
    """
    # Applying inverse transformation
    return 255 - pixel[0], 255 - pixel[1], 255 - pixel[2]


def umbral_transformation(pixel: list, threshold: int = 100) -> list:
    """Apply umbral transformation to a given pixel.

    Args:
        pixel (list): RGB values for a given pixel.

    Returns:
        list: New RGB umbral tranformed pixel values.
    """
    # Applying umbral transformation
    return tuple(0 if p < threshold else 255 for p in pixel)


def grayscale_transformation(pixel: list) -> list:
    """Apply gray scale tranformation to the given pixel.

    Args:
        pixel (list): RGB values for a given pixel.

    Returns:
        list: New RGB luminance pixel values.
    """
    # Applying grayscale transformation
    luminance = int(0.299 * pixel[0] + 0.587 * pixel[1] + 0.114 * pixel[2])
    return luminance, luminance, luminance


def copy_image(
    image_url: str, output_url: str = "output.jpg", transformation: str = None
) -> None:
    """This function copies an image and optionally applyes a transformation to it.

    Args:
        image_url (str): Input image URL.
        output_url (str, optional): Output image name. Defaults to "output.jpg".
        transformation (str, optional): Transformation to be applied to the image. Defaults to None.
    """
    # Opening the image
    image = Image.open(image_url)

    # Getting the image's width and height
    width, height = image.size

    # Creating a new image with the same size
    new_image = Image.new("RGB", (width, height))

    for x in range(width):
        for y in range(height):
            # Getting the pixel at the current position
            pixel = image.getpixel((x, y))

            # Applying the selected transformation
            if transformation == "inverse":
                transformed_pixel = inverse_transformation(pixel)
            elif transformation == "umbral":
                transformed_pixel = umbral_transformation(pixel)
            elif transformation == "grayscale":
                transformed_pixel = grayscale_transformation(pixel)
            else:
                transformed_pixel = pixel  # No transformation

            # Setting the new pixel
            new_image.putpixel((x, y), transformed_pixel)

    # Saving the new image
    new_image.save(f"./output/{transformation}_{output_url}")
    # Showing the new image
    new_image.show()


def get_histogram(image_url: str) -> list:
    """This  returns the list of 256 elements that represent the histogram of the image.

    Args:
        image_url (str): path to the image file.

    Returns:
        list: list of values that represent the histogram of the image.
    """

    # Open the image
    image = Image.open(image_url)

    # Get the image's width and height
    width, height = image.size

    # Create a list of 256 elements
    histogram = [0] * 256

    for x in range(width):
        for y in range(height):
            # Get the pixel at the current position
            pixel = image.getpixel((x, y))

            # Get the luminance of the pixel
            luminance = int(0.299 * pixel[0] + 0.587 * pixel[1] + 0.114 * pixel[2])

            # Increment the value in the histogram
            histogram[luminance] += 1

    return histogram


def plot_histogram(histogram: list) -> None:
    """This function plots the histogram of an image.

    Args:
        histogram (list): list of values that represent the histogram of the image.
    """

    # Create a list of 256 elements
    x = list(range(256))

    # Plot the histogram
    plt.bar(x, histogram, color="black", width=1)
    plt.show()


def enhance_contrast(image_url: str = "./input/image.jpg") -> None:
    """This function enhances the contrast of an image.

    Args:
        image_url (str, optional): The path to the jpg file. Defaults to 'image.jpg'.
    """

    # Open the image
    image = Image.open(image_url)

    # Get the image's width and height
    width, height = image.size

    # Create a new image with the same size
    new_image = Image.new("RGB", (width, height))

    # Get the histogram of the image
    histogram = get_histogram(image_url)

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

    # Save the new image
    new_image.save("./output/enhanced_image.jpg")

    # Show the new image
    new_image.show()


# plot_histogram(get_histogram(image_url="./input/image.jpg"))
# enhance_contrast(image_url="./input/image.jpg")
copy_image("input/vatican_night.jpg", transformation="grayscale")
