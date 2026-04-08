import torch
import torch.nn as nn
from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights


class EfficientNetV2Classifier(nn.Module):

    def __init__(self, num_signs=4):
        super().__init__()

        self.model = efficientnet_v2_s(
            weights=EfficientNet_V2_S_Weights.DEFAULT
        )

        in_features = self.model.classifier[1].in_features

        # Change output to 4 medical signs
        self.model.classifier[1] = nn.Linear(in_features, num_signs)

        # Sigmoid for multi-label output
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        x = self.model(x)

        x = self.sigmoid(x)

        return x
