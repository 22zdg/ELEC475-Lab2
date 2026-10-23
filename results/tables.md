# Lab 2 measured results

Accuracy and Macro F1 are percentages. Gaps are percentage points.
All predictive metrics below use the final trained model at the end of the specified run.
SD is the sample standard deviation across seeds (ddof=1). n=1 is not a multi-seed result.
Primary runs: LeNets 5 epochs; normalization variants and Zoo 10; BN + Aug 15.
Use the separate 15-epoch BN baseline for the controlled augmentation comparison.

## Table 7.3a - cifar10

| Architecture | Conv depth | Normalization / regularization | Aug | Params (M) | FLOPs (M) | Epoch time (s) | Inference latency (ms) |
|---|---:|---|---|---:|---:|---:|---:|
| lenet | 2 | none; None | No | 0.338486 | 7.172 | 12.40 | 0.4557 |
| lenet_modern | 2 | none; None | No | 0.446006 | 9.185 | 12.11 | 0.5250 |
| alexnet_none | 5 | none; Dropout 0.5 | No | 6.608842 | 436.695 | 23.44 | 1.1260 |
| alexnet_lrn | 5 | lrn; Dropout 0.5 | No | 6.608842 | 436.695 | 33.68 | 1.4956 |
| alexnet_bn | 5 | bn; Dropout 0.5 | No | 6.611402 | 437.966 | 26.51 | 1.1575 |
| alexnet_bn_aug | 5 | bn; Dropout 0.5 | Yes | 6.611402 | 437.966 | 28.37 | 1.1855 |
| zoo_alexnet | 5 | none; Dropout 0.5 | No | 57.044810 | 188.199 | 29.01 | 2.2456 |

Profiling device: cuda / NVIDIA GeForce RTX 3050 Laptop GPU. Latency is the median for one image. FLOPs are profiler estimates; unsupported operations may be omitted.
## Table 7.3a - tiny_imagenet

| Architecture | Conv depth | Normalization / regularization | Aug | Params (M) | FLOPs (M) | Epoch time (s) | Inference latency (ms) |
|---|---:|---|---|---:|---:|---:|---:|
| lenet | 2 | none; None | No | 0.354636 | 7.204 | 24.61 | 0.4038 |
| lenet_modern | 2 | none; None | No | 0.462156 | 9.217 | 24.97 | 0.4368 |
| alexnet_none | 5 | none; Dropout 0.5 | No | 6.803592 | 437.084 | 53.14 | 1.1585 |
| alexnet_lrn | 5 | lrn; Dropout 0.5 | No | 6.803592 | 437.084 | 70.31 | 1.4853 |
| alexnet_bn | 5 | bn; Dropout 0.5 | No | 6.806152 | 438.355 | 49.29 | 1.1792 |
| alexnet_bn_aug | 5 | bn; Dropout 0.5 | Yes | 6.806152 | 438.355 | 58.88 | 1.3020 |
| zoo_alexnet | 5 | none; Dropout 0.5 | No | 57.823240 | 189.755 | 48.44 | 2.2994 |

Profiling device: cuda / NVIDIA GeForce RTX 3050 Laptop GPU. Latency is the median for one image. FLOPs are profiler estimates; unsupported operations may be omitted.

## Table 7.3b - predictive performance

| Architecture | CIFAR-10 accuracy (%) | CIFAR-10 Macro F1 (%) | Tiny ImageNet accuracy (%) | Tiny ImageNet Macro F1 (%) |
|---|---:|---:|---:|---:|
| lenet | 52.71 ± 0.79 (n=3) | 52.30 ± 0.62 (n=3) | 9.99 ± 0.25 (n=3) | 7.86 ± 0.34 (n=3) |
| lenet_modern | 65.61 ± 1.44 (n=3) | 65.52 ± 1.38 (n=3) | 18.80 ± 0.42 (n=3) | 16.63 ± 0.57 (n=3) |
| alexnet_none | 72.73 ± 1.56 (n=3) | 72.89 ± 1.36 (n=3) | 15.26 ± 1.84 (n=3) | 12.56 ± 1.99 (n=3) |
| alexnet_lrn | 74.26 ± 1.34 (n=3) | 74.20 ± 1.50 (n=3) | 10.47 ± 8.64 (n=3) | 8.37 ± 7.24 (n=3) |
| alexnet_bn | 77.79 ± 3.16 (n=3) | 77.81 ± 3.38 (n=3) | 29.05 ± 1.18 (n=3) | 27.41 ± 1.12 (n=3) |
| alexnet_bn_aug | 81.98 ± 0.86 (n=3) | 81.89 ± 0.79 (n=3) | 33.53 ± 2.17 (n=3) | 32.18 ± 2.32 (n=3) |
| zoo_alexnet | 64.85 ± 1.32 (n=3) | 64.86 ± 1.11 (n=3) | 8.29 ± 6.77 (n=3) | 6.31 ± 5.53 (n=3) |

## Extended comparison - matched 15-epoch runs

| Dataset | Experiment | LR | Clean train accuracy (%) | Val accuracy (%) | Val Macro F1 (%) | Gap (pp) |
|---|---|---:|---:|---:|---:|---:|
| cifar10 | alexnet_bn_no_aug_15 | 0.001 | 90.82 ± 0.64 (n=3) | 80.55 ± 0.63 (n=3) | 80.50 ± 0.68 (n=3) | 10.27 ± 0.13 (n=3) |
| cifar10 | alexnet_bn_aug | 0.001 | 86.02 ± 1.39 (n=3) | 81.98 ± 0.86 (n=3) | 81.89 ± 0.79 (n=3) | 4.04 ± 0.53 (n=3) |
| cifar10 | alexnet_bn_aug_lr0005 | 0.0005 | 87.60 ± 0.72 (n=3) | 82.79 ± 0.61 (n=3) | 82.58 ± 0.42 (n=3) | 4.81 ± 0.12 (n=3) |
| tiny_imagenet | alexnet_bn_no_aug_15 | 0.001 | 44.05 ± 1.27 (n=3) | 33.23 ± 1.00 (n=3) | 31.96 ± 1.45 (n=3) | 10.82 ± 0.52 (n=3) |
| tiny_imagenet | alexnet_bn_aug | 0.001 | 38.17 ± 1.98 (n=3) | 33.53 ± 2.17 (n=3) | 32.18 ± 2.32 (n=3) | 4.64 ± 0.35 (n=3) |
| tiny_imagenet | alexnet_bn_aug_lr0005 | 0.0005 | 41.64 ± 1.41 (n=3) | 35.62 ± 0.59 (n=3) | 34.26 ± 0.58 (n=3) | 6.01 ± 0.83 (n=3) |
