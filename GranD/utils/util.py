import numpy as np
import re
import cv2


def read_pfm(path):
    """Read pfm file.

    Args:
        path (str): path to file

    Returns:
        tuple: (data, scale)
    """
    with open(path, "rb") as file:

        color = None
        width = None
        height = None
        scale = None
        endian = None

        header = file.readline().rstrip()
        if header.decode("ascii") == "PF":
            color = True
        elif header.decode("ascii") == "Pf":
            color = False
        else:
            raise Exception("Not a PFM file: " + path)

        dim_match = re.match(r"^(\d+)\s(\d+)\s$", file.readline().decode("ascii"))
        if dim_match:
            width, height = list(map(int, dim_match.groups()))
        else:
            raise Exception("Malformed PFM header.")

        scale = float(file.readline().decode("ascii").rstrip())
        if scale < 0:
            # little-endian
            endian = "<"
            scale = -scale
        else:
            # big-endian
            endian = ">"

        data = np.fromfile(file, endian + "f")
        shape = (height, width, 3) if color else (height, width)

        data = np.reshape(data, shape)
        data = np.flipud(data)

        return data, scale


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


def get_median_depth_mask_box_based(depth_image_path, bounding_box):
    depth_map, _ = read_pfm(depth_image_path)

    x1, y1, x2, y2 = bounding_box

    # Ensure coordinates are integers
    x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))

    x1, x2 = max(0, x1), min(depth_map.shape[1], x2)
    y1, y2 = max(0, y1), min(depth_map.shape[0], y2)

    # Extract the bounding box from the image
    bbox = depth_map[y1:y2, x1:x2]

    return np.median(bbox[bbox != 0])


def get_median_depth_mask_box_based_jpg(depth_image_path, bounding_box):
    depth_map = read_jpeg(depth_image_path)

    x1, y1, x2, y2 = bounding_box

    # Ensure coordinates are integers
    x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))

    x1, x2 = max(0, x1), min(depth_map.shape[1], x2)
    y1, y2 = max(0, y1), min(depth_map.shape[0], y2)

    # Extract the bounding box from the image
    bbox = depth_map[y1:y2, x1:x2]

    return round(float(np.median(bbox[bbox != 0])), 2)
