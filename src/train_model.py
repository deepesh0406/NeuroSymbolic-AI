import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms


from src.dataset import ChestXrayDataset
from src.efficientnet_model import EfficientNetV2Classifier


# ---------------- CONFIG ----------------
DEVICE = torch.device("cpu")  # Force CPU
EPOCHS = 15
BATCH_SIZE = 8     # IMPORTANT for CPU
LR = 3e-4

TRAIN_DIR = "data/chest_xray/train"
VAL_DIR = "data/chest_xray/val"
MODEL_PATH = "models/resnet50_final.pth"

os.makedirs("models", exist_ok=True)

# ---------------- TRANSFORMS ----------------
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ---------------- DATA ----------------
train_data = ChestXrayDataset(TRAIN_DIR, train_transform)
val_data = ChestXrayDataset(VAL_DIR, val_transform)

train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_data, batch_size=BATCH_SIZE)

# ---------------- MODEL ----------------
model = EfficientNetV2Classifier(num_classes=2).to(DEVICE)


# Handle imbalance
class_weights = torch.tensor([1.0, 2.0]).to(DEVICE)
criterion = nn.CrossEntropyLoss(weight=class_weights)

optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)


# ---------------- TRAINING ----------------
best_acc = 0.0

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0

    for images, labels in train_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    # -------- VALIDATION --------
    model.eval()
    correct, total = 0, 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            preds = torch.argmax(model(images), dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    acc = 100 * correct / total
    print(f"Epoch [{epoch+1}/{EPOCHS}] | Loss: {running_loss:.3f} | Val Acc: {acc:.2f}%")

    if acc > best_acc:
        best_acc = acc
        torch.save(model.state_dict(), MODEL_PATH)

print("\n✅ Training Complete")
print(f"🏆 Best Accuracy: {best_acc:.2f}%")
print(f"💾 Saved at: {MODEL_PATH}")
