"""Torchvision model adapters with output layers sized for the dataset."""
import torch
import torch.nn as nn
import torchvision.models as models

def get_torchvision_alexnet(num_classes: int = 10, pretrained: bool = False) -> nn.Module:
    """
    Load torchvision AlexNet and replace its final layer with num_classes outputs.
    Adaptive pooling keeps the original 9216 -> 4096 -> 4096 classifier for 64 x 64 input.
    """
    try:
        weights = models.AlexNet_Weights.DEFAULT if pretrained else None
        model = models.alexnet(weights=weights)
    except AttributeError:
        # Fallback for older torchvision versions
        model = models.alexnet(pretrained=pretrained)

    # Adapt classifier head for target classes
    in_features = model.classifier[6].in_features
    model.classifier[6] = nn.Linear(in_features, num_classes)
    return model


def get_torchvision_model(model_name: str = "alexnet", num_classes: int = 10, pretrained: bool = False) -> nn.Module:
    """Create an AlexNet or ResNet-18 baseline with the requested output size."""
    name = model_name.lower()
    if "alexnet" in name:
        return get_torchvision_alexnet(num_classes=num_classes, pretrained=pretrained)
    elif "resnet18" in name:
        try:
            weights = models.ResNet18_Weights.DEFAULT if pretrained else None
            model = models.resnet18(weights=weights)
        except AttributeError:
            model = models.resnet18(pretrained=pretrained)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    else:
        raise ValueError(f"Unsupported torchvision model zoo name: '{model_name}'. Choose from 'alexnet', 'resnet18'.")
