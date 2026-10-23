"""Modernized LeNet from the in-lab specification; padding 2 in both convs."""
import torch
from torch import nn


class LeNetModern(nn.Module):
    def __init__(self, num_classes=10, in_channels=3):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(in_channels, 6, 5, padding=2), nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2), nn.Conv2d(6, 16, 5, padding=2),
            nn.ReLU(inplace=True), nn.MaxPool2d(3, 2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(16 * 15 * 15, 120), nn.ReLU(inplace=True),
            nn.Linear(120, 84), nn.ReLU(inplace=True), nn.Linear(84, num_classes),
        )

    def forward(self, x):
        return self.classifier(torch.flatten(self.feature_extractor(x), 1))


if __name__ == "__main__":
    from torchinfo import summary
    summary(LeNetModern(), input_size=(1, 3, 64, 64))
