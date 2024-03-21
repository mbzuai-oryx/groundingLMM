import re
import numpy as np
import cv2


def read_image(path):
    """Read image and output RGB image (0-1).

    Args:
        path (str): path to file

    Returns:
        array: RGB image (0-1)
    """
    img = cv2.imread(path)

    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0

    return img


def write_jpeg(path, image):
    """Write depth map as a JPEG file.

    Args:
        path (str): path to file without extension
        image (array): data, float32 values
    """
    if image.dtype != np.float32:
        raise Exception("Image dtype must be float32.")

    min_val, max_val = image.min(), image.max()

    # Normalize the depth map to 8-bit
    normalized_image = ((image - min_val) / (max_val - min_val) * 255).astype(np.uint8)

    # Save the min and max depth values in the filename
    filename = f"{path}_min_{min_val:.4f}_max_{max_val:.4f}.jpg"
    cv2.imwrite(filename, normalized_image)


def read_jpeg(path):
    """Read depth map from a JPEG file.

    Args:
        path (str): path to file with extension

    Returns:
        numpy.ndarray: depth map in float32 format
    """
    # Extract min and max values from the filename
    match = re.search(r"_min_(-?\d+\.\d+)_max_(-?\d+\.\d+)\.jpg$", path)
    if not match:
        raise Exception(f"Could not extract min and max values from filename: {path}")

    min_val, max_val = map(float, match.groups())

    uint8_image = cv2.imread(path, cv2.IMREAD_UNCHANGED)  # Load as is
    float_image = uint8_image.astype(np.float32) / 255.0 * (max_val - min_val) + min_val
    return float_image
