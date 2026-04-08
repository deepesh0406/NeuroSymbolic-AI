"""
Evaluate Neuro-Symbolic AI Model
Generates Accuracy, Precision, Recall, F1, Confusion Matrix, ROC Curve
"""

import torch
import numpy as np
import os

from torch.utils.data import DataLoader
from torchvision import transforms

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_curve,
    auc
)

import matplotlib.pyplot as plt
import seaborn as sns

from dataset import ChestXrayDataset
from efficientnet_model import EfficientNetV2Classifier
from reasoning.symbolic_reasoner import symbolic_reasoning


# ================= CONFIG =================

DEVICE = torch.device("cpu")

MODEL_PATH = "models/efficientnet_signs.pth"

TEST_DIR = "data/chest_xray/test"

RESULTS_DIR = "results"

os.makedirs(RESULTS_DIR, exist_ok=True)


# ================= TRANSFORMS =================

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])


# ================= LOAD DATA =================

test_dataset = ChestXrayDataset(TEST_DIR, transform)

test_loader = DataLoader(
    test_dataset,
    batch_size=1,
    shuffle=False
)


# ================= LOAD MODEL =================

model = EfficientNetV2Classifier()

model.load_state_dict(
    torch.load(MODEL_PATH, map_location=DEVICE)
)

model.eval()


# ================= EVALUATION =================

y_true = []
y_pred = []
y_scores = []

print("\nEvaluating Neuro-Symbolic Model...\n")

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(DEVICE)

        outputs = model(images)

        probs = torch.sigmoid(outputs).cpu().numpy()[0]

        prediction, explanation = symbolic_reasoning(probs)

        score = explanation["pneumonia_score"]

        y_true.append(labels.item())
        y_pred.append(prediction)
        y_scores.append(score)


# ================= METRICS =================

accuracy = accuracy_score(y_true, y_pred)

precision = precision_score(y_true, y_pred)

recall = recall_score(y_true, y_pred)

f1 = f1_score(y_true, y_pred)

cm = confusion_matrix(y_true, y_pred)


print("===== FINAL NEURO-SYMBOLIC RESULTS =====\n")

print(f"Accuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1 Score  : {f1:.4f}")

print("\nConfusion Matrix:")
print(cm)


# ================= SAVE CONFUSION MATRIX FIGURE =================

plt.figure(figsize=(6,5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Normal","Pneumonia"],
    yticklabels=["Normal","Pneumonia"]
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")

plt.savefig(f"{RESULTS_DIR}/confusion_matrix.png", dpi=300)

plt.close()


# ================= SAVE ROC CURVE =================

fpr, tpr, thresholds = roc_curve(y_true, y_scores)

roc_auc = auc(fpr, tpr)

plt.figure()

plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
plt.plot([0,1],[0,1],'--')

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")

plt.legend()

plt.savefig(f"{RESULTS_DIR}/roc_curve.png", dpi=300)

plt.close()


print("\nSaved:")
print("results/confusion_matrix.png")
print("results/roc_curve.png")
