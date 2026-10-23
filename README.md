# ELEC 475 Lab 2

Zachary Gerhard — 20359351  
Finn Corkery — 20341468

We compared Base LeNet, Modern LeNet, AlexNet-64 with three normalization choices, and torchvision AlexNet on CIFAR-10 and Tiny ImageNet-200. We also tested augmentation and a lower learning rate using matched 15-epoch runs.

The project includes source code, all 54 completed training records, profiling measurements, logs, plots, and the report. Each configuration used seeds 42, 123 and 999. Report tables use final-epoch metrics and sample standard deviations across seeds. Checkpoints store each run's best validation-accuracy epoch.

## Project files

- `ELEC475_Lab2_Report.pdf`: report for submission.
- `ELEC475_Lab2_Report.docx`: editable report.
- `models/`, `utils/`, `configs/`: model definitions, shared utilities and configuration.
- `results/`, `plots/`, `logs/`, `commands.txt`: measurements and execution records.

## Environment

The experiments ran in Ubuntu under WSL on an NVIDIA GeForce RTX 3050 Laptop GPU, using Python 3.14.7 and PyTorch 2.14.1+cu130.

```bash
cd ~/elec475/ELEC475_Lab2_Postlab_Code/lab2_postlab
source ~/elec475/lab2/.venv/bin/activate
```

The dataset directory is `~/elec475/lab2/data`.

## Course sources

The LeNet architectures follow the in-lab manual, and AlexNet-64 follows the post-lab manual. Shared utilities were adapted from the course starter code. The Zoo baseline uses torchvision's AlexNet implementation without pretrained weights. The report cites the manuals and external papers used in the discussion.
