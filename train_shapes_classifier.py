import os
from shapes_training import get_clasify_data

# Read all the images in the shapes folder
main_path = "shapes"
images = os.listdir(main_path)

clasification_data = {}
# Get the data for each image
for image in images:
    # Get the data for the image
    r, dft_r = get_clasify_data(os.path.join(main_path, image))
