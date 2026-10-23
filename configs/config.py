"""Default model, data and training settings for Lab 2."""
import os
from dataclasses import dataclass
from typing import Tuple

@dataclass
class ExperimentConfig:
    # Model Architecture
    model_name: str = "lenet"               # Options: 'lenet', 'lenet_modern', 'alexnet', 'zoo_alexnet'
    norm_type: str = "bn"                   # Options: 'none', 'lrn', 'bn' (For AlexNet)
    pretrained: bool = False                # Pretrained weights for Zoo models
    num_classes: int = 10                   # 10 for CIFAR-10, 200 for Tiny ImageNet
    input_shape: Tuple[int, int, int] = (3, 64, 64)

    # Dataset & DataLoader
    dataset: str = "cifar10"                # Options: 'cifar10', 'tiny_imagenet'
    data_dir: str = "./data"
    batch_size: int = 128                   # Training batch size
    num_workers: int = 2
    pin_memory: bool = True
    use_augmentation: bool = True

    # Training defaults
    epochs: int = 15                        # The runner selects 5, 10 or 15 epochs by experiment.
    learning_rate: float = 1e-3             # Adam learning rate
    weight_decay: float = 1e-4              # L2 weight regularization
    seed: int = 42                          # Global random seed across Python, NumPy, PyTorch

    # Early Stopping Configuration
    early_stopping: bool = True             # Optional; disabled in the fixed-budget study.
    early_stopping_metric: str = "val_loss" # Options: 'val_loss', 'val_acc', 'val_f1'
    patience: int = 5                       # Halt training if monitored metric doesn't improve for 5 epochs
    min_delta: float = 1e-4                 # Minimum change counted as an improvement

    # Device & Hardware
    device: str = "cuda"                    # Device for this configuration

    # Checkpoint & Output Paths
    checkpoint_dir: str = "./checkpoints"
    plot_dir: str = "./plots"
    save_model_path: str = "./checkpoints/model.pth"
    plot_file_path: str = "./plots/loss_curve.png"

    def __post_init__(self):
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.plot_dir, exist_ok=True)
        os.makedirs(self.data_dir, exist_ok=True)
