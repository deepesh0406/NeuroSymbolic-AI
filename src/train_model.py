import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms

from dataset import ChestXrayDataset
from efficientnet_model import EfficientNetV2Classifier


# ---------------- CONFIG ----------------
DEVICE = torch.device("cpu")  # change to "cuda" if GPU available

EPOCHS = 15
BATCH_SIZE = 8
LR = 3e-4

TRAIN_DIR = "data/chest_xray/train"
VAL_DIR = "data/chest_xray/val"

MODEL_PATH = "models/efficientnet_signs.pth"

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

train_loader = DataLoader(train_data,
                          batch_size=BATCH_SIZE,
                          shuffle=True)

val_loader = DataLoader(val_data,
                        batch_size=BATCH_SIZE)


# ---------------- MODEL ----------------
model = EfficientNetV2Classifier().to(DEVICE)


# Multi-label loss (4 signs)
criterion = nn.BCELoss()

optimizer = optim.AdamW(model.parameters(), lr=LR)


# ---------------- TRAINING ----------------
best_acc = 0.0

for epoch in range(EPOCHS):

    model.train()

    running_loss = 0.0

    for images, labels in train_loader:

        images = images.to(DEVICE)

        # Convert pneumonia label → 4 sign labels
        # pneumonia=1 → [1,1,1,1]
        # normal=0 → [0,0,0,0]
        labels = labels.float().unsqueeze(1).repeat(1, 4).to(DEVICE)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        running_loss += loss.item()


    # ---------------- VALIDATION ----------------
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(DEVICE)

            labels = labels.float().unsqueeze(1).repeat(1, 4).to(DEVICE)

            outputs = model(images)

            preds = (outputs > 0.5).float()

            correct += (preds == labels).sum().item()

            total += labels.numel()


    acc = 100 * correct / total


    print(f"Epoch [{epoch+1}/{EPOCHS}] "
          f"| Loss: {running_loss:.4f} "
          f"| Sign Accuracy: {acc:.2f}%")



    # Save best model
    if acc > best_acc:

        best_acc = acc

        torch.save(model.state_dict(), MODEL_PATH)



# ---------------- DONE ----------------
print("\n✅ Training Complete")
print(f"🏆 Best Sign Accuracy: {best_acc:.2f}%")
print(f"💾 Model Saved at: {MODEL_PATH}")
