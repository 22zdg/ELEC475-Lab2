"""Summarize completed experiment records with means and sample standard deviations."""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
from run_postlab import LADDER, MATCHED_BASELINE, CUSTOM
from utils.visualization import plot_ablation_summary


def metric_text(rows, key):
    if not rows:
        return "Pending"
    values = [r["final"][key] for r in rows]
    if len(values) == 1:
        return f"{values[0]:.2f} (n=1)"
    return f"{mean(values):.2f} ± {stdev(values):.2f} (n={len(values)})"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--plot-dir", default="plots")
    args = parser.parse_args()
    result_dir = Path(args.results_dir)
    result_dir.mkdir(parents=True, exist_ok=True)
    groups = defaultdict(list)
    for path in sorted(result_dir.glob("*.json")):
        result = json.loads(path.read_text())
        if result.get("complete") and "final" in result:
            config = result["config"]
            experiment = config.get("experiment")
            if experiment:
                groups[(config["dataset"], experiment)].append(result)
    summary = []
    for (dataset, experiment), rows in groups.items():
        seeds = [r["config"]["seed"] for r in rows]
        if len(seeds) != len(set(seeds)):
            raise ValueError(f"Duplicate seed results for {dataset}/{experiment}")
        comparable = ["model", "norm", "epochs", "lr", "batch_size", "no_augment", "early_stopping", "data_dir", "workers"]
        first = rows[0]["config"]
        if any(any(r["config"][key] != first[key] for key in comparable) for r in rows):
            raise ValueError(f"Different settings cannot be pooled across seeds: {dataset}/{experiment}")
        record = {"dataset": dataset, "experiment": experiment, "seeds": " ".join(map(str, sorted(seeds))),
                  "n": len(rows), "epochs_requested": first["epochs"], "learning_rate": first["lr"],
                  "batch_size": first["batch_size"], "augmentation": not first["no_augment"],
                  "mean_epoch_seconds": mean(r["mean_epoch_seconds"] for r in rows),
                  "parameters": rows[0]["profile"]["trainable_params"],
                  "flops": rows[0]["profile"]["flops"]}
        for key in ["val_accuracy", "val_macro_f1", "clean_train_accuracy",
                    "generalization_gap_pp", "online_generalization_gap_pp"]:
            values = [r["final"][key] for r in rows]
            record[key + "_mean"] = mean(values)
            record[key + "_std"] = stdev(values) if len(values) > 1 else ""
        summary.append(record)
    if summary:
        with (result_dir / "summary.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(summary[0]))
            writer.writeheader()
            writer.writerows(summary)
    lines = ["# Lab 2 measured results", "",
             "Accuracy and Macro F1 are percentages. Gaps are percentage points.",
             "All predictive metrics below use the final trained model at the end of the specified run.",
             "SD is the sample standard deviation across seeds (ddof=1). n=1 is not a multi-seed result.",
             "Primary runs: LeNets 5 epochs; normalization variants and Zoo 10; BN + Aug 15.",
             "Use the separate 15-epoch BN baseline for the controlled augmentation comparison.", ""]
    for dataset in ["cifar10", "tiny_imagenet"]:
        lines.extend([f"## Table 7.3a - {dataset}", "",
                      "| Architecture | Conv depth | Normalization / regularization | Aug | Params (M) | FLOPs (M) | Epoch time (s) | Inference latency (ms) |",
                      "|---|---:|---|---|---:|---:|---:|---:|"])
        profile_path = result_dir / f"profile_{dataset}.json"
        profile_data = json.loads(profile_path.read_text()) if profile_path.exists() else None
        profiles = {r["experiment"]: r for r in profile_data["models"]} if profile_data else {}
        for experiment, model, norm, augment, epochs, lr in LADDER:
            rows = groups.get((dataset, experiment), [])
            profile = profiles.get(experiment)
            parameters = f"{profile['trainable_params'] / 1e6:.6f}" if profile else (
                f"{rows[0]['profile']['trainable_params'] / 1e6:.6f}" if rows else "Pending")
            flops = f"{profile['flops_per_image'] / 1e6:.3f}" if profile else (
                f"{rows[0]['profile']['flops'] / 1e6:.3f}" if rows else "Pending")
            epoch_time = f"{mean(r['mean_epoch_seconds'] for r in rows):.2f}" if rows else "Pending"
            latency = f"{profile['batch_latency_ms']:.4f}" if profile and profile_data["batch_size"] == 1 else "Pending (use batch 1)"
            regularization = "Dropout 0.5" if model in {"alexnet", "zoo_alexnet"} else "None"
            lines.append(f"| {experiment} | {2 if model.startswith('lenet') else 5} | {norm}; {regularization} | "
                         f"{'Yes' if augment else 'No'} | {parameters} | {flops} | {epoch_time} | {latency} |")
        if profile_data:
            lines.extend(["", f"Profiling device: {profile_data['device']} / {profile_data['device_name']}. "
                          "Latency is the median for one image. FLOPs are profiler estimates; unsupported operations may be omitted."])
        seed_rows = {}
        for experiment, *_ in LADDER:
            choices = groups.get((dataset, experiment), [])
            if choices:
                chosen = min(choices, key=lambda r: r["config"]["seed"])
                seed_rows[f"{experiment} (seed {chosen['config']['seed']})"] = {
                    "val_accs": [h["val_accuracy"] for h in chosen["history"]]}
        if seed_rows:
            plot_ablation_summary(seed_rows, str(Path(args.plot_dir) / f"ablation_summary_{dataset}.png"))
    lines.extend(["", "## Table 7.3b - predictive performance", "",
                  "| Architecture | CIFAR-10 accuracy (%) | CIFAR-10 Macro F1 (%) | Tiny ImageNet accuracy (%) | Tiny ImageNet Macro F1 (%) |",
                  "|---|---:|---:|---:|---:|"])
    for experiment, *_ in LADDER:
        c = groups.get(("cifar10", experiment), [])
        t = groups.get(("tiny_imagenet", experiment), [])
        lines.append(f"| {experiment} | {metric_text(c, 'val_accuracy')} | {metric_text(c, 'val_macro_f1')} | "
                     f"{metric_text(t, 'val_accuracy')} | {metric_text(t, 'val_macro_f1')} |")
    lines.extend(["", "## Extended comparison - matched 15-epoch runs", "",
                  "| Dataset | Experiment | LR | Clean train accuracy (%) | Val accuracy (%) | Val Macro F1 (%) | Gap (pp) |",
                  "|---|---|---:|---:|---:|---:|---:|"])
    for dataset in ["cifar10", "tiny_imagenet"]:
        for spec in [MATCHED_BASELINE, LADDER[5], CUSTOM]:
            experiment, _, _, _, _, lr = spec
            rows = groups.get((dataset, experiment), [])
            lines.append(f"| {dataset} | {experiment} | {lr} | {metric_text(rows, 'clean_train_accuracy')} | "
                         f"{metric_text(rows, 'val_accuracy')} | {metric_text(rows, 'val_macro_f1')} | "
                         f"{metric_text(rows, 'generalization_gap_pp')} |")
    (result_dir / "tables.md").write_text("\n".join(lines) + "\n")
    print(f"Summarized {sum(map(len, groups.values()))} measured runs. See {result_dir / 'tables.md'}")
    print("Unavailable measurements are labelled Pending in the summary tables.")


if __name__ == "__main__":
    main()
