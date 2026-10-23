"""Random seeds, data-loader worker initialization and sample statistics."""
import os
import random
from typing import Sequence, Tuple, Union
import numpy as np
import torch

def seed_everything(seed: int = 42, deterministic: bool = True) -> None:
    """Seed Python, NumPy and PyTorch; optionally configure deterministic cuDNN execution."""
    # 1. Python built-in hash seed and random module
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    
    # 2. NumPy random state
    np.random.seed(seed)
    
    # 3. PyTorch CPU and GPU random states
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # Covers multi-GPU setups
        
    # 4. cuDNN backend determinism
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True

def seed_worker(worker_id: int) -> None:
    """Seed each data-loader worker from its PyTorch worker seed."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def calculate_mean_std(values: Sequence[Union[int, float]]) -> Tuple[float, float]:
    """
    Return the mean and sample SD with ddof=1.
    Empty and single-value inputs return an SD of zero.
    """
    if len(values) == 0:
        return 0.0, 0.0
    if len(values) == 1:
        return float(values[0]), 0.0
    
    arr = np.array(values, dtype=np.float64)
    mean_val = float(np.mean(arr))
    std_val = float(np.std(arr, ddof=1))  # Sample SD with ddof=1
    return mean_val, std_val

def format_mean_std(values: Sequence[Union[int, float]], unit: str = "%", decimals: int = 2) -> str:
    """Format values as mean plus or minus sample SD with the requested unit."""
    mean_val, std_val = calculate_mean_std(values)
    return f"{mean_val:.{decimals}f} ± {std_val:.{decimals}f}{unit}"
