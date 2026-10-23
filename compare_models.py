"""Measure parameters, operation estimates and latency for the seven configurations."""
import argparse
import json
import platform
from pathlib import Path
import torch
from models import get_model
from utils.profiler import ModelProfiler, benchmark_inference


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["cifar10", "tiny_imagenet"], default="cifar10")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--reps", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--output", default=None, help="JSON profile output")
    args = parser.parse_args()
    if min(args.batch_size, args.reps) < 1 or args.warmup < 0:
        parser.error("batch size and reps must be positive; warmup must be nonnegative")
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    if device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA was requested but is unavailable")
    classes = 10 if args.dataset == "cifar10" else 200
    configs = [
        ("lenet", "LeNet-5", "lenet", "none"),
        ("lenet_modern", "Modernized LeNet", "lenet_modern", "none"),
        ("alexnet_none", "AlexNet, no normalization", "alexnet", "none"),
        ("alexnet_lrn", "AlexNet + LRN", "alexnet", "lrn"),
        ("alexnet_bn", "AlexNet + BatchNorm", "alexnet", "bn"),
        ("alexnet_bn_aug", "AlexNet + BN + augmentation", "alexnet", "bn"),
        ("zoo_alexnet", "PyTorch Zoo AlexNet", "zoo_alexnet", "none"),
    ]
    records = []
    print(f"Device: {device} | Dataset: {args.dataset} | Batch size: {args.batch_size}")
    print("Model | Params (M) | MACs/image (M) | FLOPs/image (M) | Batch latency (ms) | Images/s")
    for experiment, label, model_name, norm in configs:
        model = get_model(model_name, num_classes=classes, norm_type=norm).to(device)
        profile = ModelProfiler(model, (args.batch_size, 3, 64, 64), device)
        latency, throughput = benchmark_inference(
            model, (args.batch_size, 3, 64, 64), device, args.reps, args.warmup)
        record = {"experiment": experiment, "label": label, **profile.summary(),
                  "macs_per_image": profile.macs / args.batch_size,
                  "flops_per_image": profile.flops / args.batch_size,
                  "batch_latency_ms": latency, "throughput_images_per_second": throughput}
        records.append(record)
        print(f"{label} | {profile.total_params / 1e6:.6f} | {record['macs_per_image'] / 1e6:.3f} | "
              f"{record['flops_per_image'] / 1e6:.3f} | {latency:.4f} | {throughput:.1f}")
        del model, profile
    payload = {"dataset": args.dataset, "device": device,
               "device_name": torch.cuda.get_device_name(0) if device == "cuda" else platform.processor(),
               "torch": str(torch.__version__), "batch_size": args.batch_size,
               "reps": args.reps, "warmup": args.warmup, "models": records}
    output = Path(args.output or f"results/profile_{args.dataset}.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"Saved {output}")
    print("Augmentation is a training transform, so rows 5 and 6 have the same inference architecture.")
    print("FLOPs are profiler estimates; LRN, activations and other operators may be uncounted.")


if __name__ == "__main__":
    main()
