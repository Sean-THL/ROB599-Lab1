"""Part B orchestration: load scenes, train, evaluate, and save report artifacts.

Run through lab1/run_lab1.py --part b. Each voxel size gets its own freshly
initialized model, checkpoint directory, metrics and point-level predictions.
"""
from __future__ import annotations

import csv
import json
import os
from importlib.metadata import version
from pathlib import Path

import numpy as np

from lab1.dataset import load_class_names, load_segmentation_split
from lab1.segmentation import build_model, evaluate, train
from rob498.config import RunManifest, set_global_seed
from rob498.metrics import confusion_matrix


def save_json(path, value):
    """Store absent-class IoU as JSON null rather than nonstandard NaN."""
    def clean(item):
        if isinstance(item, dict):
            return {key: clean(value) for key, value in item.items()}
        if isinstance(item, (list, tuple, np.ndarray)):
            return [clean(value) for value in item]
        if isinstance(item, np.generic):
            item = item.item()
        if isinstance(item, float) and not np.isfinite(item):
            return None
        return item

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(clean(value), indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def plot_history(history, path):
    """Save learning curves; train mIoU is online, validation is end-of-epoch."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = [row["epoch"] for row in history]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(epochs, [row["train_loss"] for row in history])
    axes[0].set_ylabel("Training cross-entropy")
    axes[1].plot(epochs, [row["train_miou"] for row in history], label="Train (online)")
    axes[1].plot(epochs, [row["val_miou"] for row in history], label="Validation")
    axes[1].set_ylabel("Point-level mIoU")
    axes[1].legend()
    for axis in axes:
        axis.set_xlabel("Epoch")
        axis.grid(True)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def run_part_b(data_root, out, *, voxel_sizes=(0.05,), epochs=50, lr=0.001,
               weight_decay=0.0001, patience=10, channels=(32, 64, 128, 256),
               seed=0, task_classes=None, resume=False, evaluate_only=False):
    """Train independent models and return one metrics record per voxel size.

    Small prepared subsets are loaded into CPU RAM. This is not a lazy loader
    for the entire dataset. No pretrained weights are downloaded or loaded.
    Task classes must be chosen explicitly to match the eventual Part E task.
    """
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("Request a GPU allocation before starting Part B")
    if epochs < 1 or lr <= 0 or weight_decay < 0:
        raise ValueError("Need epochs >= 1, lr > 0 and weight_decay >= 0")
    if patience is not None and patience < 1:
        raise ValueError("patience must be positive, or None to disable early stopping")
    if not voxel_sizes or len(set(voxel_sizes)) != len(voxel_sizes):
        raise ValueError("Provide distinct voxel sizes")
    if any(not np.isfinite(size) or size <= 0 for size in voxel_sizes):
        raise ValueError("Voxel sizes must be positive and finite")

    data_root, out = Path(data_root), Path(out)
    classes = load_class_names(data_root)
    if not classes or len(set(classes)) != len(classes):
        raise ValueError("classes.txt must contain distinct, nonempty class names")
    task_classes = list(task_classes or [])
    unknown = set(task_classes) - set(classes)
    if unknown or len(set(task_classes)) != len(task_classes):
        raise ValueError(f"Task classes must be distinct names from classes.txt: {unknown}")
    task_ids = [classes.index(name) for name in task_classes]

    print(f"Loading the prepared subset from {data_root}", flush=True)
    train_scenes = load_segmentation_split(data_root, "train")
    val_scenes = load_segmentation_split(data_root, "val")
    if {scene.name for scene in train_scenes} & {scene.name for scene in val_scenes}:
        raise ValueError("Training and validation scene names overlap")
    for scene in train_scenes + val_scenes:
        # Catch a wrong class mapping rather than silently ignoring valid objects.
        if np.any(scene.segment >= len(classes)):
            raise ValueError(f"{scene.name}: label exceeds the class list; check preprocessing")
        if not len(scene) or not np.isfinite(scene.coord).all():
            raise ValueError(f"{scene.name}: empty or non-finite point cloud")
    print(f"{len(train_scenes)} train / {len(val_scenes)} validation scenes, "
          f"{len(classes)} classes; GPU: {torch.cuda.get_device_name(0)}", flush=True)
    preparation_path = data_root / "preparation.json"
    preparation = json.loads(preparation_path.read_text()) if preparation_path.exists() else None
    records = []

    for size in voxel_sizes:
        # A separate directory prevents mixing 2 cm, 5 cm and 10 cm checkpoints.
        run_dir = out / f"voxel_{size * 100:g}cm"
        run_dir.mkdir(parents=True, exist_ok=True)
        config = {
            "data_root": str(data_root.resolve()), "voxel_size_m": float(size),
            "class_names": classes, "architecture": "compact_spconv_residual_unet",
            "channels": list(channels), "in_channels": 3,
            "from_pretrained": False, "seed": seed, "lr": lr,
            "weight_decay": weight_decay, "patience": patience,
            "train_scenes": [scene.name for scene in train_scenes],
            "val_scenes": [scene.name for scene in val_scenes],
            "preparation": preparation,
        }
        config_path = run_dir / "run_config.json"
        if config_path.exists():
            if not (resume or evaluate_only):
                raise FileExistsError(f"{run_dir} already exists; enable resume or choose a new --out")
            if json.loads(config_path.read_text()) != config:
                raise ValueError(f"Run conditions changed for {run_dir}; choose a new --out")
        elif resume or evaluate_only:
            raise FileNotFoundError(f"No saved run configuration in {run_dir}")
        save_json(config_path, config)

        manifest = RunManifest(run_name=f"part_b_{size * 100:g}cm", seed=seed)
        for key, value in config.items():
            manifest.add_param(key, value)
        manifest.add_param("requested_epochs", epochs)
        manifest.add_param("trials", 1)
        manifest.add_param("torch_version", torch.__version__)
        manifest.add_param("spconv_version", version("spconv-cu121"))
        manifest.add_param("slurm_job_id", os.environ.get("SLURM_JOB_ID"))
        manifest.save(run_dir)
        save_json(run_dir / "status.json", {"status": "running"})

        try:
            set_global_seed(seed)
            model = build_model(num_classes=len(classes), channels=tuple(channels)).cuda()
            best_path, last_path = run_dir / "best.pth", run_dir / "last.pth"
            if evaluate_only:
                saved = torch.load(best_path, map_location="cpu", weights_only=False)
                model.load_state_dict(saved["model_state_dict"])
                model.best_epoch = saved["best_epoch"]
                model.training_time_s = saved["elapsed_s"]
                model.train_peak_memory_mb = saved["train_peak_memory_mb"]
                history = saved["history"]
                del saved
            else:
                model, history = train(
                    model, train_scenes, val_scenes, epochs=epochs, voxel_size=size,
                    lr=lr, weight_decay=weight_decay, patience=patience, seed=seed,
                    device="cuda", checkpoint_path=best_path,
                    resume_from=last_path if resume else None,
                    history_path=run_dir / "history.json",
                )

            result = evaluate(model, val_scenes, size)
            metrics = result.metrics(len(classes))
            per_class_iou = metrics["per_class_iou"]
            record = {
                "voxel_size_m": float(size), "miou": metrics["miou"],
                "pixel_accuracy": metrics["pixel_acc"],
                "per_class_iou": per_class_iou, "class_names": classes,
                "best_epoch": model.best_epoch, "epochs_completed": len(history),
                "training_time_s": model.training_time_s,
                "train_peak_memory_mb": model.train_peak_memory_mb,
                "inference_latency_ms": result.inference_latency_ms,
                "inference_peak_memory_mb": result.peak_memory_mb,
                "task_classes": task_classes, "task_class_ids": task_ids,
                "task_miou": result.subset_miou(task_ids, len(classes)) if task_ids else None,
                "train_scenes": config["train_scenes"], "val_scenes": config["val_scenes"],
                "seed": seed, "from_pretrained": False,
                "timing_scope": "mean forward time per scene; one warm-up; excludes preprocessing, transfers and point remapping",
            }
            save_json(run_dir / "metrics.json", record)
            save_json(run_dir / "history.json", history)
            cm = confusion_matrix(result.pred_labels, result.gt_labels, len(classes))
            with (run_dir / "per_class_iou.csv").open("w", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(["class_id", "class_name", "iou", "validation_gt_points"])
                for class_id, name in enumerate(classes):
                    iou = per_class_iou[class_id]
                    writer.writerow([class_id, name, float(iou) if np.isfinite(iou) else "", int(cm[class_id].sum())])
            np.savez_compressed(
                run_dir / "validation_predictions.npz",
                pred_labels=result.pred_labels, gt_labels=result.gt_labels,
                scene_names=np.asarray(config["val_scenes"]),
                offsets=np.cumsum([0] + [len(scene) for scene in val_scenes]),
                confusion_matrix=cm,
            )
            plot_history(history, run_dir / "training_curves.png")
            manifest.add_result("metrics", json.loads((run_dir / "metrics.json").read_text()))
            manifest.save(run_dir)
            save_json(run_dir / "status.json", {"status": "complete"})
            records.append(json.loads((run_dir / "metrics.json").read_text()))
            print(f"Saved {size * 100:g} cm results to {run_dir}; "
                  f"validation mIoU={metrics['miou']:.4f}", flush=True)
            del model, result
            torch.cuda.empty_cache()
        except Exception as error:
            save_json(run_dir / "status.json", {"status": "failed", "error": str(error)})
            raise
    return records


def main():
    """A single-model command, convenient for independent Slurm array tasks."""
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--voxel-size", type=float, default=0.05)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    parser.add_argument("--patience", type=int, default=10, help="0 disables early stopping")
    parser.add_argument("--channels", nargs="+", type=int, default=[32, 64, 128, 256])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--task-classes", nargs="+", help="Exact class names for B3")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--evaluate-only", action="store_true")
    args = parser.parse_args()
    run_part_b(
        args.data, args.out, voxel_sizes=[args.voxel_size], epochs=args.epochs,
        lr=args.lr, weight_decay=args.weight_decay,
        patience=None if args.patience == 0 else args.patience,
        channels=args.channels, seed=args.seed, task_classes=args.task_classes,
        resume=args.resume, evaluate_only=args.evaluate_only,
    )


if __name__ == "__main__":
    main()
