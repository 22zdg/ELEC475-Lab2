"""Load CIFAR-10 and the separate Tiny ImageNet training and validation splits."""
import os
import urllib.request
import zipfile
from pathlib import Path
from copy import copy
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import CIFAR10, ImageFolder
from torchvision.datasets.folder import default_loader
from .transforms import get_train_transforms, get_test_transforms
from .seed import seed_worker

CIFAR10_CLASSES = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]

TINY_IMAGENET_URL = "https://cs231n.stanford.edu/tiny-imagenet-200.zip"


class TinyImageNetValidation(Dataset):
    """Official validation images, using the training set's class indices."""
    def __init__(self, val_dir, class_to_idx, transform):
        self.transform = transform
        self.class_to_idx = dict(class_to_idx)
        self.classes = sorted(class_to_idx, key=class_to_idx.get)
        root = Path(val_dir)
        annotations = root / "val_annotations.txt"
        if annotations.is_file():
            self.samples = []
            for line in annotations.read_text().splitlines():
                name, wnid, *_ = line.split()
                if wnid not in class_to_idx:
                    raise ValueError(f"Validation class {wnid} is absent from training")
                image = root / "images" / name
                if not image.is_file():
                    raise FileNotFoundError(image)
                self.samples.append((str(image), class_to_idx[wnid]))
        else:
            organized = ImageFolder(root=str(root))
            if set(organized.classes) != set(class_to_idx):
                raise ValueError("Validation class folders do not match training")
            self.samples = [(path, class_to_idx[organized.classes[target]])
                            for path, target in organized.samples]
        if not self.samples:
            raise ValueError("No Tiny ImageNet validation images found")
        self.targets = [target for _, target in self.samples]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, target = self.samples[index]
        return self.transform(default_loader(path)), target


def get_clean_train_loader(train_loader, dataset, target_size=64):
    """Evaluate all training images with deterministic evaluation transforms."""
    clean = copy(train_loader.dataset)
    clean.transform = get_test_transforms(dataset, target_size)
    return DataLoader(clean, batch_size=train_loader.batch_size, shuffle=False,
                      num_workers=train_loader.num_workers,
                      pin_memory=train_loader.pin_memory, drop_last=False,
                      worker_init_fn=seed_worker,
                      generator=torch.Generator().manual_seed(0))

def download_and_extract_tiny_imagenet(data_dir: str = "./data"):
    target_dir = os.path.join(data_dir, "tiny-imagenet-200")
    if os.path.exists(target_dir):
        return target_dir

    os.makedirs(data_dir, exist_ok=True)
    zip_path = os.path.join(data_dir, "tiny-imagenet-200.zip")

    if not os.path.exists(zip_path):
        print(f"Downloading Tiny ImageNet-200 from {TINY_IMAGENET_URL}...")
        urllib.request.urlretrieve(TINY_IMAGENET_URL, zip_path)
        print("Download complete.")

    print(f"Extracting {zip_path}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(data_dir)
    print("Extraction complete.")
    return target_dir

def get_dataloaders(
    dataset: str = "cifar10",
    data_dir: str = "./data",
    batch_size: int = 128,
    num_workers: int = 2,
    pin_memory: bool = True,
    use_augmentation: bool = True,
    target_size: int = 64,
    seed: int = 42
):
    dataset_name = dataset.lower()

    if dataset_name == "cifar10":
        train_tf = get_train_transforms("cifar10", augment=use_augmentation, target_size=target_size)
        val_tf   = get_test_transforms("cifar10", target_size=target_size)

        train_set = CIFAR10(root=data_dir, train=True, download=True, transform=train_tf)
        val_set   = CIFAR10(root=data_dir, train=False, download=True, transform=val_tf)
        num_classes = 10
        classes = CIFAR10_CLASSES

    elif dataset_name in ["tiny_imagenet", "tinyimagenet"]:
        tiny_dir = download_and_extract_tiny_imagenet(data_dir)
        train_dir = os.path.join(tiny_dir, "train")

        train_tf = get_train_transforms("tiny_imagenet", augment=use_augmentation, target_size=target_size)
        val_tf   = get_test_transforms("tiny_imagenet", target_size=target_size)

        train_set = ImageFolder(root=train_dir, transform=train_tf)
        val_set = TinyImageNetValidation(os.path.join(tiny_dir, "val"),
                                         train_set.class_to_idx, val_tf)
        num_classes = len(train_set.classes)
        classes = train_set.classes

    else:
        raise ValueError(f"Unknown dataset '{dataset}'. Choose from 'cifar10', 'tiny_imagenet'.")

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory and torch.cuda.is_available(),
        drop_last=True,
        worker_init_fn=seed_worker,
        generator=torch.Generator().manual_seed(seed)
    )

    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory and torch.cuda.is_available(),
        drop_last=False,
        worker_init_fn=seed_worker,
        generator=torch.Generator().manual_seed(seed + 1)
    )

    return train_loader, val_loader, num_classes, classes
