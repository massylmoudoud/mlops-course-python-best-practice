import os
from PIL import Image
import torch
from torchvision import transforms, models
from torchvision.models.resnet import ResNet18_Weights


class imageData:
    def __init__(self, DIR):
        self.D = DIR

    def LoadImages(self):
        imgs = []
        for F in os.listdir(self.D):
            if F.endswith(".jpg") or F.endswith(".png"):
                imgs.append(Image.open(os.path.join(self.D, F)))
        return imgs


class imgProcess:
    def __init__(self, size: int):
        self.s = size

    def resize_and_GRAY(self, img_list: list) -> list:
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
    def __init__(self):
        self.mdl = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        self.mdl.eval()

    def Predict_Img(self, processed_images):
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
