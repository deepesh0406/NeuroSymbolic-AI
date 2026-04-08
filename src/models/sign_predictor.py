import torch
import torch.nn as nn
import torchvision.models as models

class SignPredictor(nn.Module):

    def __init__(self):

        super(SignPredictor, self).__init__()

        self.backbone = models.resnet50(pretrained=True)

        num_features = self.backbone.fc.in_features

        # remove original classifier
        self.backbone.fc = nn.Identity()

        # new classifier for 4 medical signs
        self.sign_classifier = nn.Sequential(

            nn.Linear(num_features, 256),
            nn.ReLU(),

            nn.Linear(256, 4),
            nn.Sigmoid()

        )

        self.sign_names = [
            "opacity",
            "consolidation",
            "infiltration",
            "inflammation"
        ]

    def forward(self, x):

        features = self.backbone(x)

        signs = self.sign_classifier(features)

        return signs
