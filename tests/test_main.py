import numpy as np
import PIL
from src.main import *


def test_LoadImages():
    """test if data loader returns list of PIL images"""
    image_path = "images/"
    loader = imageData(image_path)
    images = loader.LoadImages()

    assert isinstance(images[0], PIL.PngImagePlugin.PngImageFile)


def test_resize_and_GRAY():
    """test for the resize_and_GRAY method of the class imgProcess"""

    image_path = "images/"
    loader = imageData(image_path)
    images = loader.LoadImages()

    print("Original image size", images[0].size)
    image_new_size = 256
    processor = imgProcess(image_new_size)
    processed_images = processor.resize_and_GRAY(images)

    assert processed_images[0].shape == torch.Size([3, 256, 256])
