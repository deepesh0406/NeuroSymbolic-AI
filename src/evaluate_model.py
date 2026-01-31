import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc

from src.dataset import ChestXrayDataset
from src.efficientnet_model import EfficientNetV2Classifier   

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_PATH = "models/efficientnet_best.pth"   # ✅ match training
TEST_DIR = "data/chest_xray/test"
BATCH_SIZE = 16

os.makedirs("results", exist_ok=True)

# -------------------------
# transforms
# -------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],
                         [0.229,0.224,0.225])
])

dataset = ChestXrayDataset(TEST_DIR, transform=transform)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

# -------------------------
# model
# -------------------------
model = EfficientNetV2Classifier().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

all_preds = []
all_labels = []
all_probs = []

with torch.no_grad():
    for images, labels in loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        outputs = model(images)
        probs = torch.softmax(outputs, dim=1)

        preds = torch.argmax(probs, dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs[:,1].cpu().numpy())

# -------------------------
# Accuracy
# -------------------------
acc = (np.array(all_preds) == np.array(all_labels)).mean()
print(f"\n✅ Accuracy: {acc*100:.2f}%")

# -------------------------
# Confusion matrix
# -------------------------
cm = confusion_matrix(all_labels, all_preds)

plt.figure()
plt.imshow(cm)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.savefig("results/confusion_matrix.png")
plt.close()

print("\nConfusion Matrix:\n", cm)

# -------------------------
# ROC curve
# -------------------------
fpr, tpr, _ = roc_curve(all_labels, all_probs)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
plt.plot([0,1],[0,1])
plt.legend()
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.savefig("results/roc_curve.png")
plt.close()

# -------------------------
# report
# -------------------------
report = classification_report(all_labels, all_preds)
print("\nClassification Report:\n", report)

with open("results/metrics.txt","w") as f:
    f.write(report)

print("\n📁 All evaluation outputs saved in /results")
