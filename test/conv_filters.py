""" Convolution and median filters for image processing. """

from PIL import Image
import matplotlib.pyplot as plt


# ✅
def conv_filter(
    image_url: str = "./input/noisy_image.png",
    output_url: str = "./output/conv_filter_output.jpg",
) -> None:
    """Apply the convolution filter to an image.

    Args:
        image_url (str, optional): input image path. Defaults to "./input/noisy_image.png".
        output_url (str, optional): output image path. Defaults to "./output/conv_filter_output.jpg".
    """

    # Original image
    image = Image.open(image_url).convert("L")

    # Get image size
    width, height = image.size

    # Create a new image
    new_image = Image.new("L", (width, height))

    # Apply convolution filter
    for x in range(
        1, width - 1
    ):  # ignore the edge pixels for simplicity (1 to width-1 instead of 0 to width)
        for y in range(
            1, height - 1
        ):  # ignore edge pixels for simplicity (1 to height-1 instead of 0 to height)
            # Get pixel neighborhood
            neighbors = []
            for i in range(-1, 2):
                for j in range(-1, 2):
                    neighbors.append(image.getpixel((x + i, y + j)))
            # Calculate the convolution
            average = sum(neighbors) // 9  # integer division to get integer result
            # Save the new pixel in the new image
            new_image.putpixel((x, y), average)

    # Save the new image
    new_image.save(output_url)

    # Display the images
    # show_images(image_url, output_url, title="Convolution filter")
    return new_image


# ✅
def median_filter(
    image_url: str = "./input/noisy_image.png",
    output_url: str = "./output/median_filter_output.jpg",
) -> None:
    """Apply the median filter to an image.

    Args:
        image_url (str, optional): input image path. Defaults to "./input/noisy_image.png".
        output_url (str, optional): output image path. Defaults to "median_filter_output.jpg".
    """
    # Original image
    image = Image.open(image_url).convert("L")  # Convert to grayscale

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

    # Save the new image
    new_image.save(output_url)

    # Display the images
    # show_images(image_url, output_url, title="Median filter")
    return new_image


def show_images(
    original_image_url: str, result_image_url: str, title: str = "Image comparisson"
) -> None:
    """Display the original image and result image side by side.

    Args:
        original_image_url (str): path to the original image.
        result_image_url (str): path to the result image.
    """
    # Open the original image
    original_image = Image.open(original_image_url)

    # Open the result image
    result_image = Image.open(result_image_url)

    # Create a figure with two subplots
    fig, axs = plt.subplots(1, 2)

    # Set the title
    fig.suptitle(title)

    # Display the original image in the first subplot
    axs[0].imshow(original_image, cmap="gray")
    axs[0].set_title("Original Image")

    # Display the result image in the second subplot
    axs[1].imshow(result_image, cmap="gray")
    axs[1].set_title("Result Image")

    # Remove the axis labels
    for ax in axs:
        ax.axis("off")

    # Adjust the layout
    plt.tight_layout()

    # Show the figure
    plt.show()
