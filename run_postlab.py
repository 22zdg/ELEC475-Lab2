"""Run the post-lab experiments in stages and keep logs and exact commands."""
import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

# id, model, norm, augment, epochs, learning rate
LADDER = [
    ("lenet", "lenet", "none", False, 5, 0.001),
    ("lenet_modern", "lenet_modern", "none", False, 5, 0.001),
    ("alexnet_none", "alexnet", "none", False, 10, 0.001),
    ("alexnet_lrn", "alexnet", "lrn", False, 10, 0.001),
    ("alexnet_bn", "alexnet", "bn", False, 10, 0.001),
    ("alexnet_bn_aug", "alexnet", "bn", True, 15, 0.001),
    ("zoo_alexnet", "zoo_alexnet", "none", False, 10, 0.001),
]
MATCHED_BASELINE = ("alexnet_bn_no_aug_15", "alexnet", "bn", False, 15, 0.001)
CUSTOM = ("alexnet_bn_aug_lr0005", "alexnet", "bn", True, 15, 0.0005)


def experiments(stage):
    if stage == "norm":
        return LADDER[2:5]
    if stage == "augmentation":
        return [MATCHED_BASELINE, LADDER[5]]
    if stage == "custom":
        return [LADDER[5], CUSTOM]
    if stage == "ladder":
        return LADDER
    return LADDER + [MATCHED_BASELINE, CUSTOM]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["norm", "augmentation", "ladder", "custom", "all"], default="norm")
    parser.add_argument("--datasets", nargs="+", choices=["cifar10", "tiny_imagenet"], default=["cifar10"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--data-dir", default="./data")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    os.chdir(root)
    for dataset in dict.fromkeys(args.datasets):
        for seed in dict.fromkeys(args.seeds):
            for experiment, model, norm, augment, epochs, lr in experiments(args.stage):
                stem = f"{dataset}_{experiment}_s{seed}"
                result = Path("results") / (stem + ".json")
                expected = {"model": model, "norm": norm, "dataset": dataset, "seed": seed,
                            "no_augment": not augment, "epochs": epochs, "lr": lr,
                            "batch_size": args.batch_size, "experiment": experiment,
                            "data_dir": args.data_dir, "workers": args.workers,
                            "early_stopping": False, "pretrained": False,
                            "device": args.device}
                if result.exists() and not args.dry_run:
                    previous = json.loads(result.read_text())
                    if previous.get("complete") and all(previous["config"].get(k) == v for k, v in expected.items()):
                        print(f"Already complete: {stem}", flush=True)
                        continue
                    raise SystemExit(f"{result} is incomplete or has different settings. Move it aside before retrying.")
                command = [
                    sys.executable, "-u", "train.py", "--model", model, "--norm", norm,
                    "--dataset", dataset, "--data-dir", args.data_dir, "--epochs", str(epochs),
                    "--batch-size", str(args.batch_size), "--lr", str(lr), "--seed", str(seed),
                    "--workers", str(args.workers), "--experiment", experiment,
                    "--device", args.device,
                    "--save-model", f"checkpoints/{stem}.pth",
                    "--plot-file", f"plots/{stem}.png", "--results-file", str(result),
                ]
                if not augment:
                    command.append("--no-augment")
                print(shlex.join(command), flush=True)
                if args.dry_run:
                    continue
                Path("logs").mkdir(exist_ok=True)
                with Path("commands.txt").open("a") as commands:
                    commands.write(shlex.join(command) + "\n")
                with (Path("logs") / (stem + ".txt")).open("w") as log:
                    process = subprocess.Popen(command, stdout=subprocess.PIPE,
                                               stderr=subprocess.STDOUT, text=True)
                    for line in process.stdout:
                        print(line, end="", flush=True)
                        log.write(line)
                        log.flush()
                    if process.wait():
                        raise SystemExit(f"Training failed. See logs/{stem}.txt")
                if not result.exists() or not json.loads(result.read_text()).get("complete"):
                    raise SystemExit(f"Training did not produce a complete result: {result}")


if __name__ == "__main__":
    main()
