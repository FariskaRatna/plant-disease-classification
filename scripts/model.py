import torch.nn as nn
import torch
from efficientnet_pytorch import EfficientNet


class ModifiedEfficientNet(nn.Module):
    def __init__(self, num_classes, model_name="efficientnet-b0", dropout_rate=0.2):
        super(ModifiedEfficientNet, self).__init__()

        self.efficientnet = EfficientNet.from_pretrained(model_name)

        self.additional_layers = nn.Sequential(
            nn.Conv2d(1280, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
        )

        # in_features = self.efficientnet._fc.in_features
        self.additional_fc = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(64, num_classes),
        )
        self.efficientnet._fc = nn.Identity()

    def forward(self, x):
        x = self.efficientnet.extract_features(x)

        x = self.additional_layers(x)

        x = x.mean([2, 3])

        x = self.additional_fc(x)

        return x
