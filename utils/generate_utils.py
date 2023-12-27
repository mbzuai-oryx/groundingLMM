from PIL import Image, ImageFilter


def create_feathered_mask(size, border=50):
    # Create a mask with the same size as the image
    mask = Image.new('L', size, 0)

    # Calculate the inner rectangle (the non-feathered area)
    inner_rect = (border, border, size[0] - border, size[1] - border)

    # Fill the inner rectangle with white (255)
    mask.paste(255, inner_rect)

    # Apply a gaussian blur to create the feathered effect
    return mask.filter(ImageFilter.GaussianBlur(border / 2))


def center_crop(img):
    width, height = img.size
    center_x = width // 2
    center_y = height // 2
    side = min(width, height)

    # Calculate the coordinates of the box
    left = center_x - side // 2
    right = center_x + side // 2
    top = center_y - side // 2
    bottom = center_y + side // 2
    box = (left, top, right, bottom)

    # Crop the image with the box
    img_square = img.crop(box)

    return img_square, box