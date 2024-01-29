import cv2
import numpy as np


def converto_to_grayscale(url, save=False):
    # Load the image
    image = cv2.imread(url)

    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    if save:
        # Save the grayscale image
        cv2.imwrite("edge_detection_output/grayscale.bmp", gray_image)

    return gray_image


def apply_filter(url, kernel):
    """This function applies a kernel to an image, pixel by pixel.
    It only works with 3x3 kernels.

    Args:
        image (_type_): image to apply the kernel to
        kernel (_type_): kernel to apply to the image

    Returns:
        _type_: _description_
    """
    # Load the image
    image = cv2.imread(url)

    # Create a new image to store the result
    result = np.zeros(image.shape, dtype=np.uint8)

    # Apply the kernels to the image pixel by pixel
    for i in range(1, image.shape[0] - 1):
        for j in range(1, image.shape[1] - 1):
            result[i, j] = np.sum(image[i - 1 : i + 2, j - 1 : j + 2] * kernel)

    return result


def apply_prewitt(url, pixel_by_pixel=False):
    # Load the image
    image = cv2.imread(url)

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

    # Save the results
    cv2.imwrite("edge_detection_output/prewitt_90_degree_lines.bmp", result_90)
    cv2.imwrite("edge_detection_output/prewitt_0_degree_lines.bmp", result_0)
    cv2.imwrite("edge_detection_output/prewitt_45_degree_lines.bmp", result_45)
    cv2.imwrite("edge_detection_output/prewitt_-45_degree_lines.bmp", result_minus_45)

    results["90"] = result_90
    results["0"] = result_0
    results["45"] = result_45
    results["-45"] = result_minus_45

    # apply cany edge detection to each result and save the results
    for key, value in results.items():
        canny_output = cv2.Canny(value, 80, 150)
        cv2.imwrite(f"edge_detection_output/prewitt_{key}_canny.bmp", canny_output)

    # Combine the results
    result = np.zeros(result_90.shape, dtype=np.uint8)
    result = cv2.add(result, result_90)
    result = cv2.add(result, result_0)
    result = cv2.add(result, result_45)
    result = cv2.add(result, result_minus_45)

    # Save the results
    cv2.imwrite("edge_detection_output/prewitt_combined.bmp", result)

    dict_result = {
        "90": result_90,
        "0": result_0,
        "45": result_45,
        "-45": result_minus_45,
    }
    return dict_result


def apply_sobel(url, pixel_by_pixel=False):
    # Load the image
    image = cv2.imread(url)

    gain = 0
    # Define the kernels
    kernel_90 = np.array(
        [[-1, gain, 1], [-2, gain, 2], [-1, gain, 1]], dtype=np.float32
    )

    kernel_0 = np.array([[-1, -2, -1], [gain, gain, gain], [1, 2, 1]], dtype=np.float32)

    results = {}

    if pixel_by_pixel:
        # Apply the kernels to the image
        result_90 = apply_filter(image, kernel_90)
        result_0 = apply_filter(image, kernel_0)
    else:
        # Apply the kernels to the image
        result_90 = cv2.filter2D(image, -1, kernel_90)
        result_0 = cv2.filter2D(image, -1, kernel_0)

    # Save the results
    cv2.imwrite("edge_detection_output/sobel_Y_lines.bmp", result_90)
    cv2.imwrite("edge_detection_output/sobel_X_lines.bmp", result_0)

    # combine the results
    result = np.zeros(result_90.shape, dtype=np.uint8)
    result = cv2.add(result, result_90)
    result = cv2.add(result, result_0)

    results["90"] = result_90
    results["0"] = result_0

    # apply cany edge detection to each result and save the results
    for key, value in results.items():
        canny_output = cv2.Canny(value, 80, 150)
        cv2.imwrite(f"edge_detection_output/sobel_{key}_canny.bmp", canny_output)

    # Save the results
    cv2.imwrite("edge_detection_output/sobel_combined.bmp", result)

    dict_result = {"90": result_90, "0": result_0}

    return dict_result


def apply_kirsch(url, pixel_by_pixel=False):
    # Load the image
    image = cv2.imread(url)

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

    # Save the results
    cv2.imwrite("edge_detection_output/kirsch_90_degree_lines.bmp", result_90)
    cv2.imwrite("edge_detection_output/kirsch_0_degree_lines.bmp", result_0)
    cv2.imwrite("edge_detection_output/kirsch_45_degree_lines.bmp", result_45)
    cv2.imwrite("edge_detection_output/kirsch_-45_degree_lines.bmp", result_minus_45)

    results["90"] = result_90
    results["0"] = result_0
    results["45"] = result_45
    results["-45"] = result_minus_45

    # apply cany edge detection to each result and save the results
    for key, value in results.items():
        canny_output = cv2.Canny(value, 80, 150)
        cv2.imwrite(f"edge_detection_output/kirsch_{key}_canny.bmp", canny_output)

    # combine the results
    result = np.zeros(result_90.shape, dtype=np.uint8)
    result = cv2.add(result, result_90)
    result = cv2.add(result, result_0)
    result = cv2.add(result, result_45)
    result = cv2.add(result, result_minus_45)

    # Save the results
    cv2.imwrite("edge_detection_output/kirsch_combined.bmp", result)

    dict_result = {
        "90": result_90,
        "0": result_0,
        "45": result_45,
        "-45": result_minus_45,
    }

    return dict_result


if __name__ == "__main__":
    # Load the image
    IMAGE_PATH = "input/domino.jpg"
    original_image = cv2.imread(IMAGE_PATH)

    # Convert the image to grayscale
    image = converto_to_grayscale(original_image)

    # save the grayscale image
    # cv2.imwrite("edge_detection_output/grayscale.bmp", image)

    # Apply the kernels
    prewitt_90, prewitt_0, prewitt_45, prewitt_minus_45 = apply_prewitt(image)

    sobel_90, sobel_0 = apply_sobel(image)

    kirsch_90, kirsch_0, kirsch_45, kirsch_minus_45 = apply_kirsch(image)

    print("Done!")

    # Canny edge detection
    # canny_output = cv2.Canny(prewitt_0, 80, 150)

    # Save the results
    # cv2.imwrite("edge_detection_output/canny.bmp", canny_output)
