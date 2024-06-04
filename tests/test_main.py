import PIL
from src.main import ImageData, ImgProcess


def test_load_images():
    """test if data loader returns list of PIL images"""
    image_path = "images/"
    loader = ImageData(image_path)
    images = loader.load_images()

    assert isinstance(images[0], PIL.PngImagePlugin.PngImageFile)


def test_resize_and_gray():
    """test for the resize_and_gray method of the class ImgProcess"""

    image_path = "images/"
    loader = ImageData(image_path)
    images = loader.load_images()

    print("Original image size", images[0].size)
    image_new_size = 256
    processor = ImgProcess(image_new_size)
    processed_images = processor.resize_and_gray(images)

    assert processed_images[0].shape == (3, 256, 256)
