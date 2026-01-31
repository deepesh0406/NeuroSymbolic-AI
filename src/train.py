"""
Quick environment test for Neuro-Symbolic AI Project
"""

import torch
from torch import nn
import matplotlib.pyplot as plt

# 1. Check PyTorch install
print("✅ Torch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

# 2. Tiny CNN test
class TinyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 8, 3, 1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(8, 2)
    def forward(self, x):
        x = torch.relu(self.conv(x))
        x = self.pool(x).squeeze(-1).squeeze(-1)
        return self.fc(x)

model = TinyNet()
x = torch.randn(4, 3, 64, 64)
out = model(x)
print("✅ Forward pass OK, output shape:", out.shape)

# 3. Dummy training-loss curve
epochs = list(range(1, 6))
losses = [1.0, 0.8, 0.6, 0.5, 0.4]
plt.plot(epochs, losses, marker='o')
plt.title("Dummy Loss Curve Test")
plt.xlabel("Epoch")
plt.ylabel("Loss")
import os
import os

# Create an absolute path for results folder (works everywhere)
results_dir = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(results_dir, exist_ok=True)

save_path = os.path.join(results_dir, "test_loss_curve.png")
plt.savefig(save_path, dpi=150)
print(f"✅ Saved dummy plot to {save_path}")
# ✅ Save the trained model
torch.save(model.state_dict(), "models/model.pth")
print("✅ Model saved to models/model.pth")
