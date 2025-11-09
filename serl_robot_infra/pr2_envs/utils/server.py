from typing import Tuple
import base64
import numpy as np
from PIL import Image

def pil_to_base64(img: Image.Image) -> str:
    """
    Convert PIL Image to base64 string

    Args:
        img (Image.Image): PIL Image

    Returns:
        str: Base64 string
    """
    img_encoded = base64.b64encode(np.array(img).tobytes()).decode("utf-8")
    return img_encoded


def base64_to_pil(data: str, shape: Tuple[int, int]) -> Image.Image:
    """
    Convert base64 string to PIL Image

    Args:
        data (str): Base64 string

    Returns:
        Image.Image: PIL Image
    """
    img = Image.fromarray(np.frombuffer(base64.b64decode(data), np.uint8).reshape(shape))
    return img
