import os
from PIL import Image
import torch
from torchvision import transforms, models
from torchvision.models.resnet import ResNet18_Weights

"""
This is a script that loads some images and perdict their class using a pre trainned neural network wodel
"""


class imageData:
    """
    Data loader class
    """

    def __init__(self, DIR: str):
        """
        Initialize path to the folder of the images
        """
        self.D = DIR

    def LoadImages(self):
        """
        Load the images in the folder with extension jpg or png  and return them in a list
        """
        imgs = []
        for F in os.listdir(self.D):
            if F.endswith(".jpg") or F.endswith(".png"):
                imgs.append(Image.open(os.path.join(self.D, F)))
        return imgs


class imgProcess:
    """
    Image processing class
    """

    def __init__(self, size: int):
        """
        Specify the size of the images after processing
        Images are converted to square of s by s pixels
        """
        self.s = size

    def resize_and_GRAY(self, img_list: list) -> list:
        """
        Performs the processings of the images:
            Reshape the a square of specifyed size (from init )
            Convert image to gray scale (while keeping 3 channels as if it was RGB)
            Convert the image from PIL object to tensor object
            Normalize the values in the image
        Output:
            p_images: list of processed images
        """
        p_images = []
        for img in img_list:
            t = transforms.Compose(
                [
                    transforms.Resize((self.s, self.s)),
                    transforms.Grayscale(num_output_channels=3),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            )
            p_images.append(t(img))
        return p_images


class predictor:
    """
    Calls for applying the predictor network on the images
    """

    def __init__(self):
        """
        Initialize with a pre-trainned model (here resnet with 18 layers)
        """
        self.mdl = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        self.mdl.eval()

    def Predict_Img(self, processed_images: list) -> list[int]:
        """
        Predict the class of each image with the given network model
        Inputs:
            processed_images: list of processed images in the tensor form
        Outputs:
            results: list of integers representing the predicted class of each image
        """
        results = []
        for img_tensor in processed_images:
            pred = self.mdl(img_tensor.unsqueeze(0))
            results.append(torch.argmax(pred, dim=1).item())
        return results


if __name__ == "__main__":
    image_path = "images/"
    loader = imageData(image_path)
    images = loader.LoadImages()

    print("Original image size", images[0].size)
    print("Original image type", type(images[0]))
    image_new_size = 256
    processor = imgProcess(image_new_size)
    processed_images = processor.resize_and_GRAY(images)

    print(processed_images[0].shape)

    pred = predictor()
    results = pred.Predict_Img(processed_images)
    print("The predicted class is ", results)
