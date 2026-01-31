import torch
import torch.nn as nn
from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights

class EfficientNetV2Classifier(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()

        self.model = efficientnet_v2_s(
            weights=EfficientNet_V2_S_Weights.DEFAULT
        )

        in_features = self.model.classifier[1].in_features
        self.model.classifier[1] = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.model(x)
