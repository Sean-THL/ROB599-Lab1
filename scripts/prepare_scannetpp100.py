"""Prepare a small ScanNet++ top-100 subset from the course's segmentation NPZs.

Reads shared files without changing them. Writes coordinates, zero placeholder
colors and remapped labels. Actual model features must be XYZ, never label RGB.
Mapping follows https://github.com/scannetpp/scannetpp/blob/main/semantic/prep/map_semantic.py
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def class_list(path):
    names = path.read_text().splitlines()
    if not names or any(not name.strip() for name in names) or len(set(names)) != len(names):
        raise ValueError(f'Expected distinct, nonempty class names: {path}')
    return [name.strip() for name in names]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True, help='Shared data root')
    parser.add_argument('--out', type=Path, required=True, help='NEW personal data root')
    parser.add_argument('--train-scenes', type=int, default=20)
    parser.add_argument('--val-scenes', type=int, default=10)
    parser.add_argument('--exclude-scene', nargs='+', default=['a08dda47a8'])
    parser.add_argument('--max-points', type=int, default=200000,
                        help='Fixed random point subset per scene; 0 keeps all points')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    if min(args.train_scenes, args.val_scenes) < 1 or args.max_points < 0:
        parser.error('Scene counts must be positive; max-points must be nonnegative')
    if args.out.exists():
        parser.error('Output already exists; choose a new --out')
    if args.out.resolve().is_relative_to(args.source.resolve()):
        parser.error('Write outside the shared source directory')

    source = args.source / 'segmentation'
    metadata = args.source / 'scannet_plus_plus' / 'metadata' / 'semantic_benchmark'
    raw_names = class_list(source / 'classes.txt')
    top100 = class_list(metadata / 'top100.txt')
    if len(top100) != 100:
        raise ValueError(f'Expected 100 benchmark classes; found {len(top100)}')
    benchmark_ids = {name: index for index, name in enumerate(top100)}
    with (metadata / 'map_benchmark.csv').open(newline='') as handle:
        reader = csv.DictReader(handle)
        if not {'class', 'semantic_map_to'} <= set(reader.fieldnames or []):
            raise ValueError('Mapping CSV needs class and semantic_map_to columns')
        mapping = {}
        for row in reader:
            name = row['class'].strip()
            if name in mapping:
                raise ValueError(f'Duplicate mapping row: {name}')
            # Empty field means identity mapping. Literal None means ignore.
            mapping[name] = row['semantic_map_to'].strip() or name
    lookup = np.full(len(raw_names), -100, dtype=np.int64)
    mapping_rows = []
    for raw_id, name in enumerate(raw_names):
        target = mapping.get(name, name)
        lookup[raw_id] = benchmark_ids.get(target, -100)
        mapping_rows.append({'raw_id': raw_id, 'raw_name': name,
                             'mapped_name': target, 'training_id': int(lookup[raw_id])})

    # Keep the existing official scene splits. Never move validation into training.
    selected = {}
    rng = np.random.default_rng(args.seed)
    excluded = set(args.exclude_scene)
    for split, count in [('train', args.train_scenes), ('val', args.val_scenes)]:
        files = sorted(path for path in (source / split).glob('*.npz')
                       if path.stem not in excluded)
        if len(files) < count:
            raise ValueError(f'{split}: need {count} scenes, only {len(files)} available')
        selected[split] = sorted(rng.choice(files, size=count, replace=False), key=str)
    if {p.stem for p in selected['train']} & {p.stem for p in selected['val']}:
        raise ValueError('Selected training and validation scene IDs overlap')

    output = args.out / 'segmentation'
    output.mkdir(parents=True)
    (output / 'classes.txt').write_text('\n'.join(top100) + '\n')
    summary = {'dataset': 'ScanNet++ course preprocessed segmentation',
               'feature_mode': 'xyz', 'seed': args.seed, 'excluded_scenes': sorted(excluded),
               'max_points_per_scene': args.max_points,
               'sampling': 'fixed uniform random subset without replacement, independent per scene',
               'evaluation_scope': 'prepared point subset, not full-resolution benchmark',
               'source': str(args.source.resolve()), 'classes': top100, 'scenes': {}}
    for split in ('train', 'val'):
        (output / split).mkdir()
        summary['scenes'][split] = []
        for path in selected[split]:
            # These course NPZs use coord/color/segment; do not read label colors.
            with np.load(path, allow_pickle=False) as data:
                coord_key = next((key for key in ('coord', 'vtx_coords', 'points', 'xyz')
                                  if key in data), None)
                label_key = next((key for key in ('segment', 'vtx_labels', 'labels', 'semantic_gt')
                                  if key in data), None)
                if coord_key is None or label_key is None:
                    raise ValueError(f'{path}: missing coordinates or labels; keys: {data.files}')
                coords = np.asarray(data[coord_key], dtype=np.float32)
                raw = np.asarray(data[label_key], dtype=np.int64).reshape(-1)
            if coords.shape != (len(raw), 3) or not len(raw) or not np.isfinite(coords).all():
                raise ValueError(f'{path.stem}: invalid coordinates/label shapes')
            if np.any(raw >= len(lookup)):
                raise ValueError(f'{path.stem}: label exceeds raw class list')
            original_points = len(raw)
            if args.max_points and len(raw) > args.max_points:
                # Stable scene-specific seed makes the sample independent of split order.
                scene_rng = np.random.default_rng([args.seed, int(path.stem, 16)])
                indices = np.sort(scene_rng.choice(len(raw), args.max_points, replace=False))
                coords, raw = coords[indices], raw[indices]
            labels = np.full(len(raw), -100, dtype=np.int64)
            valid = raw >= 0
            labels[valid] = lookup[raw[valid]]
            counts = np.bincount(labels[labels >= 0], minlength=100)
            if not counts.sum():
                raise ValueError(f'{path.stem}: no top-100 labels remain')
            np.savez_compressed(output / split / path.name, coord=coords,
                                color=np.zeros(coords.shape, dtype=np.uint8), segment=labels)
            record = {'scene': path.stem, 'original_points': original_points,
                      'saved_points': len(labels), 'labelled_points': int(counts.sum()),
                      'class_counts': counts.tolist()}
            summary['scenes'][split].append(record)
            print(f'{split} {path.stem}: {len(labels):,} points, {counts.sum():,} labelled', flush=True)
    (args.out / 'label_mapping.json').write_text(json.dumps(mapping_rows, indent=2))
    (args.out / 'preparation.json').write_text(json.dumps(summary, indent=2))
    print(f'Prepared {args.train_scenes} train / {args.val_scenes} val scenes: {args.out}')
    print('Train with --features xyz. Placeholder colors are all zero.')


if __name__ == '__main__':
    main()
