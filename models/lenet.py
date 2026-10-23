"""Base LeNet-5 from the in-lab specification; RGB, 64 x 64 input."""
import torch
from torch import nn


class LeNet5(nn.Module):
    def __init__(self, num_classes=10, in_channels=3):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(in_channels, 6, 5), nn.Tanh(), nn.AvgPool2d(2, 2),
            nn.Conv2d(6, 16, 5), nn.Tanh(), nn.AvgPool2d(2, 2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(16 * 13 * 13, 120), nn.Tanh(),
            nn.Linear(120, 84), nn.Tanh(), nn.Linear(84, num_classes),
        )

    def forward(self, x):
        return self.classifier(torch.flatten(self.feature_extractor(x), 1))


if __name__ == "__main__":
    from torchinfo import summary
    summary(LeNet5(), input_size=(1, 3, 64, 64))
