"""Training augmentation and deterministic validation transforms for Lab 2."""
from torchvision import transforms

DATASET_STATS = {
    "cifar10": {"mean": [0.4914, 0.4822, 0.4465], "std": [0.2470, 0.2435, 0.2616]},
    "tiny_imagenet": {"mean": [0.4802, 0.4481, 0.3975], "std": [0.2764, 0.2689, 0.2821]},
}


def get_train_transforms(dataset="cifar10", augment=True, target_size=64):
    stats = DATASET_STATS[dataset.lower()]
    steps = [transforms.Resize((target_size, target_size))]
    if augment:
        steps.extend([
            transforms.RandomCrop(target_size, padding=4, padding_mode="reflect"),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        ])
    steps.extend([transforms.ToTensor(),
                  transforms.Normalize(mean=stats["mean"], std=stats["std"])])
    return transforms.Compose(steps)


def get_test_transforms(dataset="cifar10", target_size=64):
    return get_train_transforms(dataset, augment=False, target_size=target_size)
