"""
Publication-Quality GradCAM Visualization
Proper gradient-based explainability for EfficientNetV2 Neuro-Symbolic AI
"""

import torch
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt

from torchvision import transforms
from PIL import Image

from efficientnet_model import EfficientNetV2Classifier


# ================= CONFIG =================

DEVICE = torch.device("cpu")

MODEL_PATH = "models/efficientnet_signs.pth"

IMAGE_PATH = "data/chest_xray/test/PNEUMONIA/person1_virus_6.jpeg"

RESULTS_DIR = "results"

os.makedirs(RESULTS_DIR, exist_ok=True)


# ================= LOAD MODEL =================

model = EfficientNetV2Classifier()

model.load_state_dict(
    torch.load(MODEL_PATH, map_location=DEVICE)
)

model.eval()


# ================= TRANSFORM =================

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])


# ================= LOAD IMAGE =================

image = Image.open(IMAGE_PATH).convert("RGB")

input_tensor = transform(image).unsqueeze(0)

input_tensor.requires_grad = True


# ================= GRADCAM IMPLEMENTATION =================

features = []
gradients = []


def forward_hook(module, input, output):
    features.append(output)


def backward_hook(module, grad_input, grad_output):
    gradients.append(grad_output[0])


# Hook last convolution layer
target_layer = model.model.features[-1]

forward_handle = target_layer.register_forward_hook(forward_hook)

backward_handle = target_layer.register_backward_hook(backward_hook)


# Forward pass
output = model(input_tensor)

# Use highest scoring class
class_score = output.max()

# Backward pass
model.zero_grad()

class_score.backward()


# Get feature maps and gradients
feature_map = features[0].detach().numpy()[0]

gradient = gradients[0].detach().numpy()[0]


# Compute weights using gradients
weights = np.mean(gradient, axis=(1, 2))


# Compute GradCAM heatmap
heatmap = np.zeros(feature_map.shape[1:], dtype=np.float32)

for i, w in enumerate(weights):
    heatmap += w * feature_map[i]


# Apply ReLU
heatmap = np.maximum(heatmap, 0)

# Normalize
heatmap /= np.max(heatmap)


# Resize to image size
heatmap = cv2.resize(heatmap, (224, 224))

heatmap_uint8 = np.uint8(255 * heatmap)

heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)


# ================= OVERLAY =================

original = cv2.imread(IMAGE_PATH)

original = cv2.resize(original, (224, 224))

overlay = cv2.addWeighted(original, 0.6, heatmap_color, 0.4, 0)


# ================= SAVE RESULTS =================

cv2.imwrite(f"{RESULTS_DIR}/gradcam_heatmap.png", heatmap_color)

cv2.imwrite(f"{RESULTS_DIR}/gradcam_overlay.png", overlay)


# Publication-quality figure
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
plt.title("Original X-ray")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB))
plt.title("GradCAM Heatmap")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
plt.title("Explainability Overlay")
plt.axis("off")

plt.savefig(f"{RESULTS_DIR}/gradcam_figure.png", dpi=300, bbox_inches="tight")

plt.show()


# Cleanup hooks
forward_handle.remove()
backward_handle.remove()


print("\nGradCAM generation complete.")
print("Saved files:")
print("results/gradcam_heatmap.png")
print("results/gradcam_overlay.png")
print("results/gradcam_figure.png")
