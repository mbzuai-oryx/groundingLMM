
def resize_long_edge(image, target_size=384):
    # Calculate the aspect ratio
    width, height = image.size
    aspect_ratio = float(width) / float(height)

    # Determine the new dimensions
    if width > height:
        new_width = target_size
        new_height = int(target_size / aspect_ratio)
    else:
        new_width = int(target_size * aspect_ratio)
        new_height = target_size

    # Resize the image
    resized_image = image.resize((new_width, new_height), Image.ANTIALIAS)
    return resized_image
