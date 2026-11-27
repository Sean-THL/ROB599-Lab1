"""Diagnose camera axes without changing data or rerunning VGGT (CPU only)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lab1.dataset import load_eval_scene
from lab1.feedforward import align_to_ground_truth
from rob498.io3d import depth_to_points
from rob498.se3 import transform_points


def rotation_error(a, b):
    relative = a[:3, :3].T @ b[:3, :3]
    return float(np.degrees(np.arccos(np.clip((np.trace(relative) - 1) / 2, -1, 1))))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--raw-scene', type=Path, required=True)
    parser.add_argument('--predictions', type=Path, help='Existing D1 predictions.npz')
    args = parser.parse_args()
    scene = load_eval_scene(args.data)
    flip = np.diag([1., -1., -1., 1.])
    transforms_path = args.raw_scene / 'iphone' / 'nerfstudio' / 'transforms.json'
    transforms = json.loads(transforms_path.read_text())
    references = {
        Path(frame['file_path']).stem: np.asarray(frame['transform_matrix']) @ flip
        for frame in transforms.get('frames', []) + transforms.get('test_frames', [])
    }
    common = [name for name in scene.train if name in references]
    print(f'Matching exported / Nerfstudio cameras: {len(common)}')
    if common:
        # Nerfstudio is OpenGL; references above are explicitly converted to OpenCV.
        for label, extra in [('current export', np.eye(4)), ('extra Y/Z flip', flip)]:
            angles = [rotation_error(scene.pose(name) @ extra, references[name]) for name in common]
            centres = [np.linalg.norm(scene.pose(name)[:3, 3] - references[name][:3, 3]) for name in common]
            print(f'{label}: median rotation vs Nerfstudio = {np.median(angles):.3f} deg; '
                  f'median camera-centre difference = {np.median(centres):.4f} m')

    # Independently check measured training depth against the GT surface samples.
    tree = cKDTree(scene.gt_points(n=200000, seed=42))
    names = [scene.train[index] for index in np.linspace(0, len(scene.train) - 1,
                                                       min(16, len(scene.train)), dtype=int)]
    distances = {'current export': [], 'extra Y/Z flip': []}
    for name in names:
        depth = scene.depth_raw(name)[::16, ::16]
        K = scene.K.copy()
        K[:2] /= 16
        points, _ = depth_to_points(depth, K, depth_scale=scene.depth_scale, max_depth=8.0)
        if not len(points):
            continue
        for label, extra in [('current export', np.eye(4)), ('extra Y/Z flip', flip)]:
            world = transform_points(scene.pose(name) @ extra, points)
            errors, _ = tree.query(world)
            distances[label].append(errors)
    for label, groups in distances.items():
        if groups:
            errors = np.concatenate(groups)
            print(f'{label}: sensor-to-GT median = {np.median(errors):.4f} m; '
                  f'within 5 cm = {100 * np.mean(errors < 0.05):.1f}%')
        else:
            print(f'{label}: no valid sensor depths to check')

    if args.predictions:
        # Only reuse saved inference outputs. No model imports or GPU work.
        with np.load(args.predictions, allow_pickle=False) as saved:
            poses = saved['poses']
            ids = saved['view_ids'].tolist()
        gt = scene.poses(ids)
        for label, extra in [('current export', np.eye(4)), ('extra Y/Z flip', flip)]:
            scores = align_to_ground_truth(poses, gt @ extra)
            print(f'{label}: saved VGGT rotation error = {scores["mean_rotation_error_deg"]:.3f} deg; '
                  f'ATE = {scores["ate_rmse_m"]:.4f} m')
    print('Diagnostic only: no cameras, depth, or predictions were modified.')


if __name__ == '__main__':
    main()
