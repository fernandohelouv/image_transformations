"""Equalize the histogram of an image."""

from PIL import Image


def get_histogram(image):
    """Calculate the histogram of an image.

    Args:
        image (PIL.Image): Image object.

    Returns:
        list: Histogram of the image.
    """
    histogram = [0] * 256
    for pixel in image.getdata():
        if isinstance(pixel, int):
            histogram[pixel] += 1
        else:
            luminance = int(0.299 * pixel[0] + 0.587 * pixel[1] + 0.114 * pixel[2])
            histogram[luminance] += 1
    return histogram


# ✅
def equalization(image_url: str = "./input/garra2.bmp") -> None:
    """Equalize the histogram of an image.

    Args:
        image_url (str, optional): Input image url. Defaults to "./input/image.jpg".
    """
    # Open the image
    image = Image.open(image_url).convert("RGB")

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

    # Save the new image
    new_image.save("./output/equalized_image.jpg")

    # Show the new image
    # new_image.show()

    return new_image
