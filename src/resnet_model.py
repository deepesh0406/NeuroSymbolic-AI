import torch.nn as nn
from torchvision import models

class ResNet50Classifier(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()

        self.model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

        in_features = self.model.fc.in_features  # 2048

        self.model.fc = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.model(x)
