"""Export selected ScanNet++ iPhone RGB-D frames into the lab eval_scene layout.

Format references:
https://github.com/scannetpp/scannetpp/blob/main/iphone/prepare_iphone_data.py
https://scannetpp.mlsg.cit.tum.de/scannetpp/documentation
Reads measured sensor depth only; the mesh is copied for later evaluation.
"""
from __future__ import annotations

import argparse
import json
import shutil
import struct
import subprocess
import zlib
from pathlib import Path

import numpy as np
from PIL import Image


DEPTH_SHAPE = (192, 256)
FLOAT_FRAME_BYTES = 192 * 256 * 4


def depth_to_mm(values):
    """Raw float depth is metres. Preserve missing/invalid samples as zero."""
    valid = np.isfinite(values) & (values > 0) & (values < 65.535)
    return np.rint(np.where(valid, values, 0) * 1000).astype(np.uint16)


def decode_depth_frame(payload):
    """Official formats: LZ4 uint16 millimetres or raw-DEFLATE float metres."""
    # Try LZ4 FIRST, as the official toolbox does. Arbitrary LZ4 bytes can
    # sometimes also decompress as DEFLATE, but do not yield a valid frame.
    try:
        import lz4.block
    except ImportError as error:
        raise RuntimeError("Depth decoding needs: python -m pip install lz4") from error
    try:
        raw = lz4.block.decompress(payload, uncompressed_size=192 * 256 * 2)
        if len(raw) != 192 * 256 * 2:
            raise ValueError('LZ4 output is not a 192x256 uint16 depth frame')
        return np.frombuffer(raw, dtype='<u2').reshape(DEPTH_SHAPE).copy()
    except (lz4.block.LZ4BlockError, ValueError):
        pass
    try:
        raw = zlib.decompress(payload, wbits=-zlib.MAX_WBITS)
    except zlib.error as error:
        raise ValueError('Depth frame is neither valid LZ4 nor raw-DEFLATE') from error
    if len(raw) != FLOAT_FRAME_BYTES:
        raise ValueError(f'DEFLATE frame has {len(raw)} bytes; expected {FLOAT_FRAME_BYTES}')
    values = np.frombuffer(raw, dtype='<f4').reshape(DEPTH_SHAPE)
    return depth_to_mm(values)


def selected_depths(path, wanted):
    """Read only selected depths; support per-frame and whole-stream compression."""
    selected = {}
    with path.open('rb') as stream:
        header = stream.read(4)
        if len(header) != 4:
            raise ValueError('depth.bin is truncated')
        first_size = struct.unpack('<I', header)[0]
        # Per-frame records start with a little-endian compressed byte count.
        per_frame = 0 < first_size <= min(path.stat().st_size - 4, FLOAT_FRAME_BYTES * 2)
        stream.seek(0)
        if per_frame:
            for frame_id in range(max(wanted) + 1):
                header = stream.read(4)
                if len(header) != 4:
                    raise ValueError(f'No depth record for frame {frame_id}')
                size = struct.unpack('<I', header)[0]
                if size < 1 or stream.tell() + size > path.stat().st_size:
                    raise ValueError(f'Invalid depth record at frame {frame_id}')
                if frame_id in wanted:
                    try:
                        selected[frame_id] = decode_depth_frame(stream.read(size))
                    except ValueError as error:
                        raise ValueError(f'Depth frame {frame_id}: {error}') from error
                else:
                    stream.seek(size, 1)
        else:
            # Older recordings compress all float32 depth frames in one stream.
            # Decode incrementally rather than loading the entire sequence in RAM.
            decoder = zlib.decompressobj(wbits=-zlib.MAX_WBITS)
            pending = bytearray()
            frame_id = 0
            while frame_id <= max(wanted):
                compressed = stream.read(65536)
                if not compressed:
                    raise ValueError('Global depth stream ended before selected frames')
                pending.extend(decoder.decompress(compressed))
                complete = len(pending) // FLOAT_FRAME_BYTES
                for offset in range(complete):
                    begin = offset * FLOAT_FRAME_BYTES
                    if frame_id in wanted:
                        values = np.frombuffer(pending[begin:begin + FLOAT_FRAME_BYTES],
                                               dtype='<f4').reshape(DEPTH_SHAPE)
                        selected[frame_id] = depth_to_mm(values)
                    frame_id += 1
                del pending[:complete * FLOAT_FRAME_BYTES]
    return selected


def extract_rgb(video, indices, destination):
    """Decode once with FFmpeg, saving only requested original frame numbers."""
    expression = '+'.join(f'eq(n,{index})' for index in indices)
    subprocess.run([
        'ffmpeg', '-nostdin', '-hide_banner', '-loglevel', 'error', '-noautorotate',
        '-i', str(video), '-vf', f"select='{expression}'", '-vsync', '0',
        '-start_number', '0', str(destination / '%06d.png'),
    ], check=True)
    files = sorted(destination.glob('*.png'))
    if len(files) != len(indices):
        raise ValueError(f'FFmpeg produced {len(files)} images, expected {len(indices)}')
    return files


def warp_to_camera(image, source_K, target_K, size, nearest=False):
    """Resample into one shared camera K, compensating per-frame focal variation."""
    # For target pixel q, its source location is source_K @ inv(target_K) @ q.
    mapping = source_K @ np.linalg.inv(target_K)
    coefficients = tuple(mapping[:2].reshape(-1))
    return image.transform(
        size, Image.Transform.AFFINE, coefficients,
        resample=Image.Resampling.NEAREST if nearest else Image.Resampling.BILINEAR,
        fillcolor=0,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scene-dir', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True, help='NEW personal data folder')
    parser.add_argument('--max-frames', type=int, default=200)
    parser.add_argument('--frame-step', type=int, default=10)
    parser.add_argument('--test-every', type=int, default=8)
    parser.add_argument('--width', type=int, default=640)
    parser.add_argument('--height', type=int, default=480)
    parser.add_argument('--pose-convention', choices=('opengl', 'opencv'), default='opencv',
                        help='Camera axes of aligned_pose; inspect sensor-to-mesh alignment')
    args = parser.parse_args()
    if args.max_frames < 2 or args.frame_step < 1 or args.test_every < 2:
        parser.error('Need max-frames >= 2, frame-step >= 1 and test-every >= 2')
    if min(args.width, args.height) < 1:
        parser.error('Image dimensions must be positive')
    if args.out.exists():
        parser.error('Output exists; choose a NEW --out to avoid mixing scenes')
    if args.out.resolve().is_relative_to(args.scene_dir.resolve()):
        parser.error('Output must be outside the raw scene directory')
    if not shutil.which('ffmpeg') or not shutil.which('ffprobe'):
        parser.error('ffmpeg and ffprobe are required on PATH')
    iphone = args.scene_dir / 'iphone'
    mesh = args.scene_dir / 'scans' / 'mesh_aligned_0.05.ply'
    required = [iphone / 'rgb.mkv', iphone / 'depth.bin',
                iphone / 'pose_intrinsic_imu.json', mesh]
    for path in required:
        if not path.is_file():
            parser.error(f'Missing file: {path}')
    records = json.loads((iphone / 'pose_intrinsic_imu.json').read_text())

    # Use aligned poses in the mesh frame, never the unaligned raw ARKit pose.
    frames = []
    for name, record in sorted(records.items()):
        if not name.startswith('frame_') or 'aligned_pose' not in record:
            continue
        index = int(name.split('_')[-1])
        pose = np.asarray(record['aligned_pose'], dtype=float)
        K = np.asarray(record['intrinsic'], dtype=float)
        if (pose.shape != (4, 4) or K.shape != (3, 3)
                or not np.isfinite(pose).all() or not np.isfinite(K).all()):
            continue
        if not np.allclose(pose[3], [0, 0, 0, 1], atol=1e-4):
            continue
        if not np.allclose(pose[:3, :3].T @ pose[:3, :3], np.eye(3), atol=0.005):
            continue
        if not np.isclose(np.linalg.det(pose[:3, :3]), 1, atol=0.005):
            continue
        if K[0, 0] <= 0 or K[1, 1] <= 0:
            continue
        frames.append((index, name, pose, K))
    frames = frames[::args.frame_step]
    if len(frames) > args.max_frames:
        chosen = np.linspace(0, len(frames) - 1, args.max_frames, dtype=int)
        frames = [frames[index] for index in chosen]
    if len(frames) < args.test_every:
        parser.error('Too few valid aligned poses for this split; reduce frame-step/test-every')

    # RGB resolution is read from the actual video, not assumed from intrinsics.
    probe = subprocess.run([
        'ffprobe', '-v', 'error', '-select_streams', 'v:0',
        '-show_entries', 'stream=width,height', '-of', 'json', str(iphone / 'rgb.mkv'),
    ], check=True, capture_output=True, text=True)
    dimensions = json.loads(probe.stdout)['streams'][0]
    source_width, source_height = dimensions['width'], dimensions['height']
    # Apply the same pixel-coordinate scaling to image and camera matrix.
    rgb_scale = np.diag([args.width / source_width, args.height / source_height, 1])
    K_shared = rgb_scale @ frames[0][3]
    output_size = (args.width, args.height)
    depths = selected_depths(iphone / 'depth.bin', {frame[0] for frame in frames})
    out = args.out / 'eval_scene'
    (out / 'images').mkdir(parents=True)
    (out / 'depth').mkdir()
    temporary = args.out / '_selected_rgb'
    temporary.mkdir()
    rgb_files = extract_rgb(iphone / 'rgb.mkv', [frame[0] for frame in frames], temporary)
    cameras = {'K': K_shared.tolist(), 'width': args.width, 'height': args.height,
               'depth_scale': 1000.0, 'frames': [], 'train': [], 'test': []}
    flip = np.diag([1., -1., -1., 1.]) if args.pose_convention == 'opengl' else np.eye(4)
    depth_scale = np.diag([256 / source_width, 192 / source_height, 1])
    for position, ((index, name, pose, K), rgb_file) in enumerate(zip(frames, rgb_files)):
        with Image.open(rgb_file) as image:
            rgb = warp_to_camera(image.convert('RGB'), K, K_shared, output_size)
            rgb.save(out / 'images' / f'{name}.png')
        # Measured iPhone depth is already aligned to RGB, at lower resolution.
        # NEAREST preserves missing pixels and depth discontinuities.
        depth_image = Image.fromarray(depths[index])
        depth_image = warp_to_camera(depth_image, depth_scale @ K, K_shared, output_size, nearest=True)
        depth_image.save(out / 'depth' / f'{name}.png')
        cameras['frames'].append({'name': name, 'T_wc': (pose @ flip).tolist()})
        split = 'test' if (position + 1) % args.test_every == 0 else 'train'
        cameras[split].append(name)
        print(f'Exported {name} ({split})', flush=True)
    shutil.copy2(mesh, out / 'mesh.ply')
    (out / 'cameras.json').write_text(json.dumps(cameras, indent=2))
    manifest = dict(vars(args))
    manifest.update(scene_id=args.scene_dir.name, pose_source='aligned_pose',
                    depth_source='measured iPhone LiDAR, not GT mesh rendering',
                    source_resolution=[source_width, source_height],
                    intrinsics_handling='warp per-frame RGB/depth into first selected frame K',
                    train=cameras['train'], test=cameras['test'])
    (args.out / 'scene_preparation.json').write_text(json.dumps(manifest, indent=2, default=str))
    shutil.rmtree(temporary)
    print(f'Prepared {out}: {len(cameras["train"])} train / {len(cameras["test"])} held-out views')
    print('Before fitting: check training depth points against the mesh for camera-axis alignment.')


if __name__ == '__main__':
    main()
