"""Lab 1 Part B -- Learned segmentation.  [25 pts]

B1 permits starting from released weights, and says: if you do, SAY SO and
report both fine-tuned and zero-shot numbers. Do that. A fine-tuned number
presented as if trained from scratch is the kind of thing that unravels in
office hours.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class SegmentationResult:
    """Predictions plus everything the summary table needs."""

    pred_labels: np.ndarray            # (N,) per-point predicted class
    gt_labels: np.ndarray              # (N,) per-point ground truth, -1 = ignore
    inference_latency_ms: float
    peak_memory_mb: float
    voxel_size_m: float
    from_pretrained: bool = False      # B1: you must report this

    def metrics(self, num_classes: int) -> dict:
        """PROVIDED. mIoU and per-class IoU via the shared metrics module."""
        from rob498.metrics import confusion_matrix, iou_from_confusion

        cm = confusion_matrix(self.pred_labels, self.gt_labels, num_classes)
        return iou_from_confusion(cm)

    def subset_miou(self, class_ids: list[int], num_classes: int) -> float:
        """PROVIDED. mIoU restricted to the classes your Part E task needs. [B3]

        B3 asks whether mIoU was a good proxy for what you actually needed. Run
        this with your task's classes and compare against the full mIoU. If they
        diverge -- and they usually do, because mIoU is dominated by large easy
        classes like floor and wall -- that divergence IS the answer to B3.
        """
        iou = self.metrics(num_classes)["per_class_iou"]
        sel = iou[class_ids]
        return float(np.nanmean(sel))


def voxelize(points: np.ndarray, voxel_size: float):
    """PROVIDED. Quantize points to a voxel grid.

    Returns (coords (M, 3) int32, inverse (N,) int) where `inverse` maps each
    original point to its voxel, so you can push per-voxel predictions back to
    per-point labels for evaluation.

    Evaluate at POINT level, not voxel level. Voxel-level mIoU at 10 cm looks
    much better than it is, and B2's whole finding -- which classes degrade
    fastest as resolution coarsens -- disappears if you evaluate in the same
    coarsened space you predicted in.
    """
    pts = np.asarray(points, dtype=float)
    coords = np.floor(pts / voxel_size).astype(np.int32)
    uniq, inverse = np.unique(coords, axis=0, return_inverse=True)
    # numpy 2.0.0 returned `inverse` as (N, 1) when `axis` is given; flatten so
    # the documented (N,) shape holds on every numpy version.
    return uniq, inverse.reshape(-1)


# --------------------------------------------------------------------------- #
# YOU WRITE THESE
# --------------------------------------------------------------------------- #
def build_model(architecture: str = "minkowski", num_classes: int = 20, **kwargs):
    """Construct a compact MinkowskiNet-style sparse U-Net. [B1]

    Supported architecture: "minkowski" (implemented with spconv).
    Optional kwargs: in_channels=3, channels=(32, 64, 128, 256).
    Returns an untrained torch.nn.Module; move it to CUDA before inference.

    Call model(features, coordinates, batch_size=None):
      features: (M, in_channels) floating voxel features.
      coordinates: (M, 4) int32 [batch, x, y, z], unique and nonnegative.
      output: (M, num_classes) raw logits in input voxel order.

    This compact residual U-Net is not an exact MinkUNet34 implementation
    and does not load released weights. See SparseUNet for the input contract.
    """
    if architecture != "minkowski":
        raise ValueError(
            f"Unsupported architecture {architecture!r}; choose 'minkowski'. "
            "PTv3 is not implemented."
        )
    
    unknown = set(kwargs) - {"in_channels", "channels"}

    if unknown:
        raise TypeError(f"Unexpected model options: {sorted(unknown)}")
    
    # Keep the provided CPU utilities importable without GPU dependencies.
    try:
        if __package__:
            from .sparse_unet import SparseUNet
        else:
            from sparse_unet import SparseUNet
    except ModuleNotFoundError as exc:
        if exc.name and exc.name.split(".")[0] in {"torch", "spconv", "cumm"}:
            raise ImportError(
                "build_model requires PyTorch and spconv. Activate the rob498 "
                "environment created from environment.yml."
            ) from exc
        raise
    return SparseUNet(num_classes=num_classes, **kwargs)



def train(model, train_loader, val_loader, epochs: int = 50, **kwargs):
    """Train and return the fitted model plus a training log.  [B1, 12 pts]

    Log per-epoch train and val mIoU and keep the curve for your report. If val
    mIoU peaks at epoch 12 and decays after, report the peak and say you
    early-stopped -- do not quietly report the last epoch.

    For simplicity, each loader is a list of SegScene objects from
    load_segmentation_split(), not a collated PyTorch DataLoader. One scene
    is one training step. Both training and validation use original points.

    Options: voxel_size=0.05, lr=0.001, weight_decay=0.0001, patience=10,
    seed=0, device=None, checkpoint_path=None. Set patience=None to disable
    early stopping. Seed model initialization BEFORE calling build_model.
    Returns (model_with_best_validation_weights, list_of_epoch_records).
    """
    import torch
    from pathlib import Path
    from rob498.metrics import confusion_matrix, iou_from_confusion

    voxel_size = kwargs.pop("voxel_size", 0.05)
    lr = kwargs.pop("lr", 0.001)
    weight_decay = kwargs.pop("weight_decay", 0.0001)
    patience = kwargs.pop("patience", 10)
    seed = kwargs.pop("seed", 0)
    device = kwargs.pop("device", None)
    checkpoint_path = kwargs.pop("checkpoint_path", None)
    if kwargs:
        raise TypeError(f"Unknown training options: {sorted(kwargs)}")
    if epochs < 1 or (patience is not None and patience < 1):
        raise ValueError("epochs and patience must be positive")

    # Lists can be reused for every epoch; an exhausted generator cannot.
    train_scenes = list(train_loader)
    val_scenes = list(val_loader)
    if not train_scenes or not val_scenes:
        raise ValueError("Training and validation both need at least one scene")

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = torch.nn.CrossEntropyLoss()
    rng = np.random.default_rng(seed)
    num_classes = model.num_classes
    history = []
    best_miou = -float("inf")
    best_state = None
    best_epoch = 0
    epochs_without_improvement = 0

    for epoch in range(1, epochs + 1):
        model.train()
        train_cm = np.zeros((num_classes, num_classes), dtype=np.int64)
        total_loss = 0.0
        total_points = 0

        # Shuffle scenes, keeping each scene's points together.
        for index in rng.permutation(len(train_scenes)):
            scene = train_scenes[index]
            features, coordinates, inverse, labels = _prepare_scene(
                scene, voxel_size, device
            )
            valid = (labels >= 0) & (labels < num_classes)
            n_valid = int(valid.sum().item())
            if n_valid == 0:
                continue

            optimizer.zero_grad(set_to_none=True)
            voxel_logits = model(features, coordinates)
            # Points sharing a voxel share its scores, but keep their own labels.
            point_logits = voxel_logits[inverse[valid]]
            loss = criterion(point_logits, labels[valid])
            loss.backward()
            optimizer.step()

            predictions = voxel_logits.detach().argmax(dim=1)[inverse].cpu().numpy()
            train_cm += confusion_matrix(predictions, scene.segment, num_classes)
            total_loss += loss.item() * n_valid
            total_points += n_valid
            # Release this scene's graph and tensors before loading the next one.
            del features, coordinates, inverse, labels, valid
            del voxel_logits, point_logits, loss

        if total_points == 0:
            raise ValueError("Training scenes contain no valid class labels")

        # Train mIoU uses predictions collected during updates; validation uses
        # the fixed weights at the end of the epoch, with BatchNorm in eval mode.
        train_miou = iou_from_confusion(train_cm)["miou"]
        optimizer.zero_grad(set_to_none=True)
        validation = evaluate(model, val_scenes, voxel_size)
        val_miou = validation.metrics(num_classes)["miou"]
        if not np.isfinite(val_miou):
            raise ValueError("Validation mIoU is undefined; check validation labels")
        history.append({
            "epoch": epoch,
            "train_loss": total_loss / total_points,
            "train_miou": train_miou,
            "val_miou": val_miou,
        })
        print(f"Epoch {epoch:03d} | loss {history[-1]['train_loss']:.4f} | "
              f"train mIoU {train_miou:.4f} | val mIoU {val_miou:.4f}")

        if val_miou > best_miou:
            best_miou = val_miou
            best_epoch = epoch
            # Copy to CPU so the saved weights do not consume extra GPU memory.
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if patience is not None and epochs_without_improvement >= patience:
            print(f"Early stopping at epoch {epoch}")
            break

    model.load_state_dict(best_state)
    model.eval()
    model.trained_voxel_size_m = float(voxel_size)
    model.best_epoch = best_epoch
    model.best_val_miou = best_miou
    print(f"Restored epoch {best_epoch}, validation mIoU {best_miou:.4f}")

    if checkpoint_path is not None:
        checkpoint_path = Path(checkpoint_path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model_state_dict": best_state,
            "best_epoch": best_epoch,
            "best_val_miou": best_miou,
            "voxel_size_m": float(voxel_size),
            "num_classes": num_classes,
            "in_channels": model.in_channels,
            "channels": model.channels,
            "from_pretrained": bool(getattr(model, "from_pretrained", False)),
            "seed": seed,
            "lr": lr,
            "weight_decay": weight_decay,
            "history": history,
        }, checkpoint_path)
    return model, history


def evaluate(model, loader, voxel_size: float) -> SegmentationResult:
    """Run inference and package a SegmentationResult.  [B1]

    Measure latency and peak GPU memory here rather than estimating them later:
    ``torch.cuda.max_memory_allocated`` after ``reset_peak_memory_stats``, and
    ``torch.cuda.synchronize()`` before you stop the clock. Without the
    synchronize you are timing kernel launches, not kernels, and the number will
    be absurdly good.

    loader contains unbatched SegScene objects. Latency is mean forward-pass
    time per scene, excluding voxelization, transfers, and point remapping.
    The first scene gets an untimed warm-up. Memory is peak PyTorch-allocated
    memory during inference, including all live tensors on this device, in MiB.
    """
    import time
    import torch

    device = next(model.parameters()).device
    use_cuda = device.type == "cuda"
    was_training = model.training
    model.eval()
    all_predictions = []
    all_labels = []
    elapsed_ms = []
    peak_memory_mb = 0.0

    try:
        with torch.no_grad():
            for scene in loader:
                features, coordinates, inverse, labels = _prepare_scene(
                    scene, voxel_size, device
                )
                if not elapsed_ms:
                    warmup = model(features, coordinates)
                    del warmup
                if use_cuda:
                    torch.cuda.synchronize(device)
                    torch.cuda.reset_peak_memory_stats(device)

                start = time.perf_counter()
                logits = model(features, coordinates)
                if use_cuda:
                    torch.cuda.synchronize(device)
                elapsed_ms.append((time.perf_counter() - start) * 1000.0)

                if use_cuda:
                    peak_memory_mb = max(
                        peak_memory_mb,
                        torch.cuda.max_memory_allocated(device) / (1024 ** 2),
                    )
                # Always score original points, not the coarser voxel grid.
                predictions = logits.argmax(dim=1)[inverse].cpu().numpy()
                all_predictions.append(predictions)
                all_labels.append(np.asarray(scene.segment).reshape(-1))
                del features, coordinates, inverse, labels, logits
    finally:
        model.train(was_training)

    if not all_predictions:
        raise ValueError("Evaluation needs at least one scene")
    return SegmentationResult(
        pred_labels=np.concatenate(all_predictions),
        gt_labels=np.concatenate(all_labels),
        inference_latency_ms=float(np.mean(elapsed_ms)),
        peak_memory_mb=peak_memory_mb,
        voxel_size_m=float(voxel_size),
        from_pretrained=bool(getattr(model, "from_pretrained", False)),
    )


def voxel_size_ablation(models, loader, voxel_sizes=(0.02, 0.05, 0.10), *,
                        plot_path="results/part_b/voxel_size_ablation.png") -> list:
    """Ablate voxel size; plot mIoU and latency against it.  [B2, 7 pts]

    `models` maps each voxel size to a model TRAINED at that voxel size. Running
    one model at voxel sizes it was not trained at measures train/test mismatch
    -- its kernels' receptive field in metres changes with the voxel -- rather
    than what the resolution costs, so train once per size. Returns one record
    per voxel size: mIoU, per-class IoU, and latency.

    B2 asks which classes degrade fastest as resolution coarsens and what they
    have in common. Sort per-class IoU by the drop from 2 cm to 10 cm and look
    at the top of that list before you write the explanation -- the answer is a
    property of the objects, and it will be obvious once you see them together.

    Saves a two-panel mIoU/latency plot to plot_path; use None to skip plotting.
    Records use Python lists for per-class IoU so they can be saved as JSON.
    """
    scenes = list(loader)
    voxel_sizes = tuple(voxel_sizes)
    if not voxel_sizes:
        raise ValueError("Provide at least one voxel size")
    if len({id(models[size]) for size in voxel_sizes}) != len(voxel_sizes):
        raise ValueError("Use a separate model instance for each voxel size")

    # Keep only one compared model on the GPU during its memory measurement.
    devices = {
        size: next(models[size].parameters()).device for size in voxel_sizes
    }
    records = []
    try:
        for size in voxel_sizes:
            models[size].cpu()
        for size in voxel_sizes:
            model = models[size].to(devices[size])
            trained_size = getattr(model, "trained_voxel_size_m", size)
            if not np.isclose(trained_size, size):
                raise ValueError(f"Model for {size} m was trained at {trained_size} m")
            result = evaluate(model, scenes, size)
            metrics = result.metrics(model.num_classes)
            records.append({
                "voxel_size_m": float(size),
                "miou": metrics["miou"],
                "per_class_iou": metrics["per_class_iou"].tolist(),
                "inference_latency_ms": result.inference_latency_ms,
                "peak_memory_mb": result.peak_memory_mb,
            })
            model.cpu()
    finally:
        for size in voxel_sizes:
            models[size].to(devices[size])

    if plot_path is not None:
        from pathlib import Path
        import matplotlib.pyplot as plt

        sizes_cm = [row["voxel_size_m"] * 100 for row in records]
        fig, axes = plt.subplots(1, 2, figsize=(9, 4))
        axes[0].plot(sizes_cm, [row["miou"] for row in records], "o-")
        axes[0].set_ylabel("Point-level mIoU")
        axes[1].plot(sizes_cm, [row["inference_latency_ms"] for row in records], "o-")
        axes[1].set_ylabel("Mean forward time per scene (ms)")
        for axis in axes:
            axis.set_xlabel("Voxel size (cm)")
            axis.set_xticks(sizes_cm)
            axis.grid(True)
        fig.tight_layout()
        plot_path = Path(plot_path)
        plot_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(plot_path, dpi=150)
        plt.close(fig)
    return records


def _prepare_scene(scene, voxel_size, device):
    """Turn one SegScene into model inputs and the original point labels.

    This simple pipeline uses mean RGB per voxel as its three input features.
    No points are cropped or sampled. Large scenes may need a later cropping
    strategy if full-scene training exceeds the GPU memory budget.
    """
    import torch

    if not np.isfinite(voxel_size) or voxel_size <= 0:
        raise ValueError("voxel_size must be positive and finite")
    if len(scene.coord) == 0:
        raise ValueError("Cannot prepare an empty scene")

    coordinates, inverse = voxelize(scene.coord, voxel_size)
    num_voxels = len(coordinates)

    # Average the colors of all points assigned to each voxel.
    colors = np.asarray(scene.color, dtype=np.float32) / 255.0
    features = np.zeros((num_voxels, 3), dtype=np.float32)
    np.add.at(features, inverse, colors)
    counts = np.bincount(inverse, minlength=num_voxels)
    features /= counts[:, None]

    # Translate integer indices without changing voxel membership or row order.
    coordinates = coordinates - coordinates.min(axis=0)
    batch_column = np.zeros((num_voxels, 1), dtype=np.int32)
    coordinates = np.concatenate([batch_column, coordinates], axis=1)

    return (
        torch.as_tensor(features, dtype=torch.float32, device=device),
        torch.as_tensor(coordinates, dtype=torch.int32, device=device),
        torch.as_tensor(inverse, dtype=torch.long, device=device),
        torch.as_tensor(scene.segment, dtype=torch.long, device=device).reshape(-1),
    )
