"""This module equalizes the histogram of an image."""


def equialization() -> None:
    """This functions equalizes the histogram of an image.

    Args:
        image_url (str, optional): Input image url. Defaults to "./input/image.jpg".
    """

    # Get the histogram of the image
    histogram = [790, 1023, 850, 656, 329, 245, 122, 81]
    cumulative = 0
    n = len(histogram) - 1

    for luminance in histogram:
        result = round(luminance / sum(histogram), 2)
        cumulative = cumulative + result
        y1 = round(cumulative * n, 0)
        print(result, cumulative, y1)


equialization()


# """This function equalizes the histogram of an image."""

# # Get the histogram of the image
# histogram = [790, 1023, 850, 656, 329, 245, 122, 81]
# total_luminance = sum(histogram)
# cumulative = 0
# N = len(histogram) - 1

# # Calculate the equalized histogram
# equalized_histogram = [round(luminance / total_luminance, 2) for luminance in histogram]
# cumulative_histogram = [cumulative := cumulative + luminance for luminance in equalized_histogram]
# equalized_values = [round(cumulative * N, 0) for cumulative in cumulative_histogram]

# # Print the results
# for result, cumulative, y1 in zip(equalized_histogram, cumulative_histogram, equalized_values):
#     print(result, cumulative, y1)
