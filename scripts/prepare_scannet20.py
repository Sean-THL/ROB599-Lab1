"""Prepare a small ScanNet20 subset for lab1/dataset.py, one scene at a time.

Read shared scans without modifying them. Write to a NEW personal output folder.
Only the cleaned mesh and its labelled counterpart are needed for Part B.
Official conventions:
https://github.com/ScanNet/ScanNet/blob/master/README.md
https://github.com/ScanNet/ScanNet/blob/master/BenchmarkScripts/3d_evaluation/evaluate_semantic_label.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from urllib.request import urlopen

import numpy as np
from plyfile import PlyData


# Official ScanNet20 NYU40 IDs, in the same order as the names below.
CLASS_IDS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39]
CLASS_NAMES = [
    "wall", "floor", "cabinet", "bed", "chair", "sofa", "table", "door",
    "window", "bookshelf", "picture", "counter", "desk", "curtain",
    "refrigerator", "shower curtain", "toilet", "sink", "bathtub", "otherfurniture",
]
SPLIT_BASE_URL = "https://raw.githubusercontent.com/ScanNet/ScanNet/master/Tasks/Benchmark"


def prepare_scene(scans_dir: Path, scene_name: str, output: Path) -> dict:
    """Combine true RGB and NYU40 labels, then remap to 0..19 / -1."""
    scene_dir = scans_dir / scene_name
    mesh = PlyData.read(scene_dir / f"{scene_name}_vh_clean_2.ply")["vertex"].data
    labelled = PlyData.read(
        scene_dir / f"{scene_name}_vh_clean_2.labels.ply"
    )["vertex"].data

    # Use RGB from the original mesh. The labelled PLY's colors visualize classes
    # and must NEVER be fed into the model as input features (label leakage).
    coord = np.column_stack([mesh[axis] for axis in ("x", "y", "z")]).astype(np.float32)
    color = np.column_stack([mesh[c] for c in ("red", "green", "blue")]).astype(np.uint8)
    label_coord = np.column_stack([labelled[axis] for axis in ("x", "y", "z")])
    if coord.shape != label_coord.shape or not np.allclose(coord, label_coord, atol=1e-5, rtol=0):
        raise ValueError(f"{scene_name}: mesh vertices and labelled vertices do not align")
    if not len(coord) or not np.isfinite(coord).all():
        raise ValueError(f"{scene_name}: empty mesh or non-finite coordinates")

    # Official .labels.ply already stores NYU40 IDs. No TSV conversion is needed.
    raw_labels = np.asarray(labelled["label"], dtype=np.int64)
    segment = np.full(len(raw_labels), -1, dtype=np.int64)
    for training_id, nyu40_id in enumerate(CLASS_IDS):
        segment[raw_labels == nyu40_id] = training_id
    if not np.any(segment >= 0):
        raise ValueError(f"{scene_name}: no points belong to ScanNet20 classes")

    # Preserve every vertex and its original coordinates in metres.
    np.savez_compressed(output, coord=coord, color=color, segment=segment)
    print(f"{scene_name}: {len(coord):,} points, {(segment >= 0).sum():,} labelled",
          flush=True)
    return {"scene": scene_name, "points": len(coord),
            "valid_points": int((segment >= 0).sum())}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True,
                        help="ScanNet dataset directory containing scans/")
    parser.add_argument("--out", type=Path, required=True, help="NEW personal output directory")
    parser.add_argument("--train-scenes", type=int, default=10)
    parser.add_argument("--val-scenes", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--splits-dir", type=Path,
                        help="Optional directory with scannetv2_train.txt and scannetv2_val.txt; "
                             "otherwise download official split lists")
    args = parser.parse_args()
    if args.train_scenes < 1 or args.val_scenes < 1:
        parser.error("Choose at least one training and one validation scene")
    scans_dir = args.source / "scans"
    if not scans_dir.is_dir():
        parser.error(f"Scans directory not found: {scans_dir}")
    if args.out.resolve().is_relative_to(args.source.resolve()):
        parser.error("Output must be outside the source dataset directory")
    if args.out.exists():
        parser.error("Output already exists. Choose a new folder to avoid mixing subsets")

    rng = np.random.default_rng(args.seed)
    split_texts = {}
    selected = {}
    unavailable = {}
    for split, count in (("train", args.train_scenes), ("val", args.val_scenes)):
        filename = f"scannetv2_{split}.txt"
        if args.splits_dir is None:
            with urlopen(f"{SPLIT_BASE_URL}/{filename}", timeout=60) as response:
                text = response.read().decode("utf-8")
        else:
            text = (args.splits_dir / filename).read_text()
        split_texts[split] = text
        scene_names = sorted(set(text.split()))
        available = []
        unavailable[split] = []
        for name in scene_names:
            scene_dir = scans_dir / name
            required = [scene_dir / f"{name}_vh_clean_2.ply",
                        scene_dir / f"{name}_vh_clean_2.labels.ply"]
            if all(path.is_file() for path in required):
                available.append(name)
            else:
                unavailable[split].append(name)
        print(f"{split}: {len(available)} usable scenes, "
              f"{len(unavailable[split])} unavailable; selecting {count}", flush=True)
        if count > len(available):
            parser.error(f"Not enough available {split} scenes for the requested subset")
        selected[split] = sorted(rng.choice(available, size=count, replace=False).tolist())
    if set(selected["train"]) & set(selected["val"]):
        parser.error("Training and validation scene selections overlap")

    args.out.mkdir(parents=True)
    seg_dir = args.out / "segmentation"
    seg_dir.mkdir()
    (seg_dir / "classes.txt").write_text("\n".join(CLASS_NAMES) + "\n")
    manifest = {
        "dataset": "ScanNet v2, ScanNet20 labels",
        "source": str(args.source.resolve()),
        "seed": args.seed,
        "nyu40_ids_in_training_order": CLASS_IDS,
        "class_names": CLASS_NAMES,
        "ignore_label": -1,
        "split_base_url": SPLIT_BASE_URL,
        "selected_scenes": selected,
        "unavailable_scenes": unavailable,
        "scenes": {},
    }
    for split in ("train", "val"):
        folder = seg_dir / split
        folder.mkdir()
        # Save both original split membership and the exact selected subset.
        (args.out / f"scannetv2_{split}.txt").write_text(split_texts[split])
        (args.out / f"selected_{split}.txt").write_text("\n".join(selected[split]) + "\n")
        manifest["scenes"][split] = [
            prepare_scene(scans_dir, name, folder / f"{name}.npz")
            for name in selected[split]
        ]
    (args.out / "preparation.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Prepared dataset root: {args.out}", flush=True)


if __name__ == "__main__":
    main()
