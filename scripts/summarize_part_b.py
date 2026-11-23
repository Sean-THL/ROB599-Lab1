"""Build B1/B2 tables and B3 subset scores from completed runs, without training.

python scripts/summarize_part_b.py --runs /path/to/results/part_b_seed42
Add --task-classes with exact class names once your Part E classes are chosen.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lab1.part_b import save_json
from rob498.report import Table, write_report_bundle


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, required=True,
                        help="Directory containing voxel_2cm/, voxel_5cm/ and voxel_10cm/")
    parser.add_argument("--out", type=Path, help="Default: <runs>/summary")
    parser.add_argument("--task-classes", nargs="+", help="Exact class names, e.g. chair table")
    args = parser.parse_args()
    out = args.out or args.runs / "summary"
    rows, configs, manifests = [], [], []
    for path in args.runs.glob("voxel_*cm/metrics.json"):
        status = json.loads((path.parent / "status.json").read_text())
        if status["status"] != "complete":
            print(f"Skipping incomplete run: {path.parent}")
            continue
        rows.append(json.loads(path.read_text()))
        configs.append(json.loads((path.parent / "run_config.json").read_text()))
        manifests.append(json.loads((path.parent / "manifest.json").read_text()))
    if not rows:
        parser.error("No completed metrics.json files found under voxel_*cm/")

    # A comparison must use the same scenes, classes and training settings.
    reference = {key: value for key, value in configs[0].items() if key != "voxel_size_m"}
    for config, manifest in zip(configs, manifests):
        if {key: value for key, value in config.items() if key != "voxel_size_m"} != reference:
            parser.error("Run conditions differ beyond voxel size; do not combine these runs")
        if manifest["params"]["requested_epochs"] != manifests[0]["params"]["requested_epochs"]:
            parser.error("Requested epoch budgets differ; keep short initial runs separate")

    rows.sort(key=lambda row: row["voxel_size_m"])
    classes = rows[0]["class_names"]
    task_classes = args.task_classes or []
    if set(task_classes) - set(classes):
        parser.error("Some task class names are not in classes.txt")
    if len(set(task_classes)) != len(task_classes):
        parser.error("Task class names must be distinct")
    task_ids = [classes.index(name) for name in task_classes]
    for row in rows:
        iou = np.asarray(row["per_class_iou"], dtype=float)
        selected = iou[task_ids]
        row["task_classes"] = task_classes
        row["task_class_ids"] = task_ids
        row["task_miou"] = float(np.nanmean(selected)) if np.isfinite(selected).any() else None

    out.mkdir(parents=True, exist_ok=True)
    table = Table("Part B -- segmentation ablation", [
        "voxel (cm)", "mIoU", "task-class mIoU", "best epoch",
        "epochs completed", "forward ms/scene", "inference peak MiB", "training seconds",
    ])
    for row in rows:
        table.add_row([
            row["voxel_size_m"] * 100, row["miou"], row["task_miou"],
            row["best_epoch"], row["epochs_completed"], row["inference_latency_ms"],
            row["inference_peak_memory_mb"], row["training_time_s"],
        ])
    table.conditions = (
        f"seed={reference['seed']}; trials=1; trained from scratch; "
        f"train scenes={reference['train_scenes']}; validation scenes={reference['val_scenes']}; "
        f"GPUs={sorted(set(m['gpu'] or 'unknown' for m in manifests))}; "
        f"maximum epochs={manifests[0]['params']['requested_epochs']}; "
        f"channels={reference['channels']}; lr={reference['lr']}; "
        f"weight_decay={reference['weight_decay']}; patience={reference['patience']}"
    )
    table.notes = (
        "Point-level evaluation on the selected validation subset, used for checkpoint "
        "selection; this is not a held-out test result or a full-benchmark score. "
        "Classes absent from both prediction and truth have undefined IoU, excluded from mIoU. "
        "Latency is mean forward time per scene after one warm-up, excluding data preparation "
        "and transfers. Memory is PyTorch peak allocated GPU memory, not occupancy query memory."
    )
    if task_classes:
        table.notes += f" B3 task classes: {task_classes}; justify these using Part E."
    else:
        table.notes += " B3 is not configured: rerun with --task-classes after choosing Part E classes."
    write_report_bundle(out, [table])
    save_json(out / "ablation.json", rows)
    save_json(out / "conditions.json", {"configs": configs, "manifests": manifests})

    with (out / "per_class_ablation.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["class_id", "class_name"] + [f"IoU_{r['voxel_size_m'] * 100:g}cm" for r in rows])
        for index, name in enumerate(classes):
            writer.writerow([index, name] + [row["per_class_iou"][index] for row in rows])

    # Rank class degradation only where both endpoint IoUs are defined.
    endpoints = {round(row["voxel_size_m"], 6): row for row in rows}
    if 0.02 in endpoints and 0.10 in endpoints:
        fine = endpoints[0.02]["per_class_iou"]
        coarse = endpoints[0.10]["per_class_iou"]
        drops = [
            (name, fine[i], coarse[i], fine[i] - coarse[i])
            for i, name in enumerate(classes)
            if fine[i] is not None and coarse[i] is not None
        ]
        drops.sort(key=lambda row: row[3], reverse=True)
        with (out / "class_iou_drop_2_to_10cm.csv").open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["class_name", "iou_2cm", "iou_10cm", "iou_drop"])
            writer.writerows(drops)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sizes_cm = [row["voxel_size_m"] * 100 for row in rows]
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    axes[0].plot(sizes_cm, [row["miou"] for row in rows], "o-")
    axes[0].set_ylabel("Validation point-level mIoU")
    axes[1].plot(sizes_cm, [row["inference_latency_ms"] for row in rows], "o-")
    axes[1].set_ylabel("Mean forward time per scene (ms)")
    for axis in axes:
        axis.set_xlabel("Voxel size (cm)")
        axis.set_xticks(sizes_cm)
        axis.grid(True)
    fig.tight_layout()
    fig.savefig(out / "voxel_size_ablation.png", dpi=150)
    plt.close(fig)
    if not {0.02, 0.05, 0.10}.issubset(endpoints):
        print("B2 is incomplete until independent 2 cm, 5 cm and 10 cm runs finish.")
    if not task_classes:
        print("B3 awaits an explicit task-class selection; no retraining is needed.")
    print(f"Saved tables and plots to {out}")


if __name__ == "__main__":
    main()
