"""Five-stage AlexNet for 64 x 64 RGB input (Lab 2, Sections 3-4)."""
import torch
from torch import nn


class AlexNet64(nn.Module):
    def __init__(self, num_classes=10, in_channels=3, norm_type="bn"):
        super().__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.norm_type = norm_type.lower()
        if self.norm_type not in {"none", "lrn", "bn"}:
            raise ValueError("norm_type must be none, lrn, or bn")

        layers = []
        stages = [(in_channels, 64, 5, 2, 2), (64, 192, 5, 1, 2),
                  (192, 384, 3, 1, 1), (384, 384, 3, 1, 1),
                  (384, 256, 3, 1, 1)]
        for stage, (cin, cout, kernel, stride, padding) in enumerate(stages):
            layers.append(nn.Conv2d(cin, cout, kernel, stride, padding))
            # BatchNorm precedes ReLU in all five stages.
            if self.norm_type == "bn":
                layers.append(nn.BatchNorm2d(cout))
            layers.append(nn.ReLU(inplace=True))
            # LRN follows ReLU in the first two stages only.
            if self.norm_type == "lrn" and stage < 2:
                layers.append(nn.LocalResponseNorm(5, alpha=1e-4, beta=0.75, k=2.0))
            if stage in (0, 1, 4):
                layers.append(nn.MaxPool2d(3, stride=2))
        self.features = nn.Sequential(*layers)
        self.classifier = nn.Sequential(
            nn.Dropout(0.5), nn.Linear(256 * 3 * 3, 1024), nn.ReLU(inplace=True),
            nn.Dropout(0.5), nn.Linear(1024, 1024), nn.ReLU(inplace=True),
            nn.Linear(1024, num_classes),
        )

    def forward(self, x):
        return self.classifier(torch.flatten(self.features(x), start_dim=1))


if __name__ == "__main__":
    from torchinfo import summary
    summary(AlexNet64(), input_size=(1, 3, 64, 64), mode="eval")
