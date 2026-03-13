from pathlib import Path
from PIL import Image


def get_image_dimensions(image_path: Path) -> tuple[int, int]:
    """Get image width and height in pixels.

    Args:
        image_path: Path to image file

    Returns:
        (width, height) tuple
    """
    with Image.open(image_path) as img:
        return img.width, img.height
