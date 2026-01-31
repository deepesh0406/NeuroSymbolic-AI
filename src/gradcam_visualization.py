import torch
import cv2
import numpy as np
from torchvision import transforms
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from src.efficientnet_model import EfficientNetV2Classifier

DEVICE = "cpu"
MODEL_PATH = "models/efficientnet_best.pth"
IMAGE_PATH = "data/chest_xray/test/PNEUMONIA/person134_bacteria_642.jpeg"

model = EfficientNetV2Classifier().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

img = Image.open(IMAGE_PATH).convert("RGB")
input_tensor = transform(img).unsqueeze(0)

target_layers = [model.model.features[-1]]
cam = GradCAM(model=model, target_layers=target_layers,)

grayscale_cam = cam(input_tensor=input_tensor,
                    targets=[ClassifierOutputTarget(1)])[0]

rgb_img = np.array(img.resize((224,224))) / 255.0
visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

cv2.imwrite("results/gradcam/gradcam_pneumonia.png",
             cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR))

print("✅ Grad-CAM saved in results/gradcam/")
