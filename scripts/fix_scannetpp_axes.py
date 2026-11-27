"""Correct the old ScanNet++ export's extra camera-axis flip in a NEW folder.

Only for exports whose scene_preparation.json records pose_convention=opengl.
The a08dda47a8 sensor-to-mesh diagnostic established that aligned_pose is OpenCV.
RGB/depth and train/test IDs are copied unchanged. No inference or fitting runs.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True, help='Old prepared data root')
    parser.add_argument('--out', type=Path, required=True, help='NEW corrected data root')
    args = parser.parse_args()
    if args.out.exists():
        parser.error('Output exists; choose a new --out')
    if args.out.resolve().is_relative_to(args.data.resolve()):
        parser.error('Output must not be inside the original data folder')
    preparation = json.loads((args.data / 'scene_preparation.json').read_text())
    if preparation.get('pose_convention') != 'opengl' or preparation.get('pose_source') != 'aligned_pose':
        parser.error('Expected an old aligned_pose export with pose_convention=opengl; refusing another flip')
    if preparation.get('scene_id') != 'a08dda47a8':
        parser.error('This correction was verified for a08dda47a8 only; check other scenes separately')
    cameras = json.loads((args.data / 'eval_scene' / 'cameras.json').read_text())
    flip = np.diag([1., -1., -1., 1.])
    for frame in cameras['frames']:
        pose = np.asarray(frame['T_wc'], dtype=float)
        if pose.shape != (4, 4) or not np.isfinite(pose).all():
            raise ValueError(f'Invalid pose for {frame["name"]}')
        frame['T_wc'] = (pose @ flip).tolist()
    preparation['pose_convention'] = 'opencv'
    preparation['axis_correction'] = {
        'copied_from': str(args.data.resolve()),
        'operation': 'remove extra Y/Z camera-axis flip; translations unchanged',
        'reason': 'measured-depth alignment and saved VGGT orientation diagnostic',
    }
    shutil.copytree(args.data, args.out)
    (args.out / 'eval_scene' / 'cameras.json').write_text(json.dumps(cameras, indent=2))
    (args.out / 'scene_preparation.json').write_text(json.dumps(preparation, indent=2))
    print(f'Corrected {len(cameras["frames"])} cameras in {args.out}')
    print('Original data retained. RGB, depth, intrinsics and frame splits are unchanged.')
    print('Use this new data root for C, D and E. Old C fits initialized with wrong axes must be refitted.')


if __name__ == '__main__':
    main()
