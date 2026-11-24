"""Export one raw ScanNet scene into the lab's eval_scene/ layout (CPU only).

Reads ScanNet version-4 .sens files using the official format documentation:
https://github.com/ScanNet/ScanNet/tree/master/SensReader
https://github.com/ScanNet/ScanNet/blob/master/SensReader/c%2B%2B/src/sensorData.h

Only selected frames are decompressed. The source scene is never modified.
The copied GT mesh is for evaluation, not Gaussian initialization.
"""
from __future__ import annotations

import argparse
import io
import json
import shutil
import struct
import sys
import zlib
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rob498.se3 import inverse


def read_values(stream, format_string):
    """Read little-endian numbers, failing clearly for a truncated input."""
    size = struct.calcsize("<" + format_string)
    data = stream.read(size)
    if len(data) != size:
        raise ValueError("The .sens file is truncated")
    return struct.unpack("<" + format_string, data)


def read_matrix(stream):
    return np.array(read_values(stream, "16f"), dtype=float).reshape(4, 4)


def index_sensor_file(path):
    """Read calibration and frame offsets without loading the full recording."""
    frames = []
    invalid_poses = 0
    file_size = path.stat().st_size
    with path.open("rb") as stream:
        if read_values(stream, "I")[0] != 4:
            raise ValueError("Only ScanNet .sens version 4 is supported")
        name_size = read_values(stream, "Q")[0]
        if name_size > file_size:
            raise ValueError("Invalid sensor-name length")
        sensor_name = stream.read(name_size).decode("utf-8", errors="replace")
        K_color, E_color = read_matrix(stream), read_matrix(stream)
        K_depth, E_depth = read_matrix(stream), read_matrix(stream)
        color_type, depth_type = read_values(stream, "ii")
        color_width, color_height, depth_width, depth_height = read_values(stream, "4I")
        depth_scale = read_values(stream, "f")[0]
        frame_count = read_values(stream, "Q")[0]
        for index in range(frame_count):
            pose = read_matrix(stream)
            timestamps = read_values(stream, "QQ")
            color_bytes, depth_bytes = read_values(stream, "QQ")
            color_offset = stream.tell()
            depth_offset = color_offset + color_bytes
            next_frame = depth_offset + depth_bytes
            if next_frame > file_size:
                raise ValueError(f"Recording is truncated at frame {index}")
            stream.seek(next_frame)
            # ScanNet can contain failed tracking poses, often stored as -inf.
            valid_pose = (
                np.isfinite(pose).all()
                and np.allclose(pose[3], [0, 0, 0, 1], atol=1e-4)
                and np.allclose(pose[:3, :3].T @ pose[:3, :3], np.eye(3), atol=2e-3)
                and np.isclose(np.linalg.det(pose[:3, :3]), 1.0, atol=2e-3)
            )
            if not valid_pose:
                invalid_poses += 1
                continue
            frames.append({
                "index": index, "pose": pose, "timestamps": timestamps,
                "color_offset": color_offset, "color_bytes": color_bytes,
                "depth_offset": depth_offset, "depth_bytes": depth_bytes,
            })
    if color_type not in (0, 1, 2) or depth_type not in (0, 1):
        raise ValueError("Unsupported color/depth compression; expected RGB/PNG/JPEG and raw/zlib")
    if min(color_width, color_height, depth_width, depth_height) < 1:
        raise ValueError("Invalid image dimensions in recording")
    if not np.isfinite(depth_scale) or depth_scale <= 0:
        raise ValueError("Invalid depth scale in recording")
    if not all(np.isfinite(matrix).all() for matrix in (K_color, E_color, K_depth, E_depth)):
        raise ValueError("Non-finite sensor calibration")
    # Official saveToPointCloud maps depth-camera points directly with each pose,
    # and maps depth -> color using E_depth when the color extrinsic is identity.
    # Refuse unusual calibration rather than silently guess its pose convention.
    if not np.allclose(E_color, np.eye(4), atol=1e-5):
        raise ValueError("Nonidentity color extrinsic: inspect sensor conventions before export")
    if (not np.allclose(E_depth[3], [0, 0, 0, 1], atol=1e-4)
            or not np.allclose(E_depth[:3, :3].T @ E_depth[:3, :3], np.eye(3), atol=2e-3)
            or not np.isclose(np.linalg.det(E_depth[:3, :3]), 1, atol=2e-3)):
        raise ValueError("Depth-to-color calibration is not a rigid transform")
    calibration = {
        "sensor_name": sensor_name, "K_color": K_color, "K_depth": K_depth,
        "T_color_depth": E_depth, "color_type": color_type, "depth_type": depth_type,
        "color_width": color_width, "color_height": color_height,
        "depth_width": depth_width, "depth_height": depth_height,
        "depth_scale": depth_scale, "total_frames": frame_count,
        "invalid_poses": invalid_poses,
    }
    return calibration, frames


def align_depth_to_color(depth, calibration, K_output, width, height):
    """Project measured depth to the resized RGB camera, keeping the nearest Z.

    This is sensor-depth registration, not depth rendered from the GT mesh.
    Pixels without a measured projection remain zero; no holes are invented.
    """
    v, u = np.nonzero(depth)
    z = depth[v, u].astype(float) / calibration["depth_scale"]
    pixels = np.column_stack([u, v, np.ones(len(u))])
    rays = pixels @ np.linalg.inv(calibration["K_depth"][:3, :3]).T
    points_depth = rays * z[:, None]
    transform = calibration["T_color_depth"]
    points_color = points_depth @ transform[:3, :3].T + transform[:3, 3]
    valid = (np.isfinite(points_color).all(axis=1) & (points_color[:, 2] > 0)
             & (points_color[:, 2] * calibration["depth_scale"] <= 65535))
    points_color = points_color[valid]
    projected = points_color @ K_output.T
    uv = projected[:, :2] / projected[:, 2:3]
    uv = np.rint(uv).astype(np.int64)
    inside = (uv[:, 0] >= 0) & (uv[:, 0] < width) & (uv[:, 1] >= 0) & (uv[:, 1] < height)
    uv, points_color = uv[inside], points_color[inside]
    nearest = np.full(width * height, np.inf)
    np.minimum.at(nearest, uv[:, 1] * width + uv[:, 0], points_color[:, 2])
    aligned = np.zeros(width * height, dtype=np.uint16)
    observed = np.isfinite(nearest)
    aligned[observed] = np.clip(
        np.rint(nearest[observed] * calibration["depth_scale"]), 1, 65535
    ).astype(np.uint16)
    return aligned.reshape(height, width)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-dir", type=Path, required=True,
                        help="Raw directory such as raw/scene0000_00")
    parser.add_argument("--out", type=Path, required=True,
                        help="Data root; creates <out>/eval_scene/")
    parser.add_argument("--max-frames", type=int, default=60)
    parser.add_argument("--test-every", type=int, default=5,
                        help="Every Nth selected frame is held out; all others train")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--mesh", type=Path,
                        help="Default: the scene's high-quality _vh_clean.ply")
    args = parser.parse_args()
    if args.max_frames < 3 or args.test_every < 2 or min(args.width, args.height) < 1:
        parser.error("Need max-frames >= 3, test-every >= 2 and positive image dimensions")
    scene_name = args.scene_dir.name
    recording = args.scene_dir / f"{scene_name}.sens"
    mesh = args.mesh or args.scene_dir / f"{scene_name}_vh_clean.ply"
    target = args.out / "eval_scene"
    if not recording.is_file() or not mesh.is_file():
        parser.error(f"Need both {recording} and {mesh}")
    if target.exists():
        parser.error("eval_scene already exists; choose a new --out to avoid overwriting")
    if target.resolve().is_relative_to(args.scene_dir.resolve()):
        parser.error("Output must be outside the raw scene directory")

    calibration, frames = index_sensor_file(recording)
    count = min(args.max_frames, len(frames))
    if count < 3:
        parser.error("Recording has fewer than three valid camera poses")
    selected = [frames[i] for i in np.linspace(0, len(frames) - 1, count, dtype=int)]
    if args.test_every > count:
        parser.error("test-every exceeds the available selected frames; no held-out view would exist")
    sx, sy = args.width / calibration["color_width"], args.height / calibration["color_height"]
    # Match PIL's pixel-centre resize mapping, including principal-point shifts.
    resize = np.array([[sx, 0, (sx - 1) / 2], [0, sy, (sy - 1) / 2], [0, 0, 1]])
    K_output = resize @ calibration["K_color"][:3, :3]
    if K_output[0, 0] <= 0 or K_output[1, 1] <= 0:
        parser.error("Color camera has invalid focal lengths")

    (target / "images").mkdir(parents=True)
    (target / "depth").mkdir()
    metadata = {
        "K": K_output.tolist(), "width": args.width, "height": args.height,
        "depth_scale": calibration["depth_scale"], "frames": [], "train": [], "test": [],
    }
    with recording.open("rb") as stream:
        for order, frame in enumerate(selected):
            name = f"frame_{frame['index']:06d}"
            stream.seek(frame["color_offset"])
            color_bytes = stream.read(frame["color_bytes"])
            if calibration["color_type"] == 0:
                image = Image.frombytes("RGB", (calibration["color_width"], calibration["color_height"]), color_bytes)
            else:
                with Image.open(io.BytesIO(color_bytes)) as decoded:
                    image = decoded.convert("RGB")
            if image.size != (calibration["color_width"], calibration["color_height"]):
                raise ValueError(f"{name}: decoded RGB dimensions disagree with calibration")
            image = image.resize((args.width, args.height), Image.Resampling.BILINEAR)
            image.save(target / "images" / f"{name}.png")

            stream.seek(frame["depth_offset"])
            depth_bytes = stream.read(frame["depth_bytes"])
            if calibration["depth_type"] == 1:
                depth_bytes = zlib.decompress(depth_bytes)
            depth = np.frombuffer(depth_bytes, dtype="<u2").reshape(
                calibration["depth_height"], calibration["depth_width"]
            )
            aligned = align_depth_to_color(depth, calibration, K_output, args.width, args.height)
            if not np.any(aligned):
                raise ValueError(f"{name}: no registered depth; inspect sensor calibration")
            Image.fromarray(aligned).save(target / "depth" / f"{name}.png")

            # Raw poses map depth-camera -> world. Export RGB-camera -> world.
            # Keep the raw mesh/world frame: do not apply axisAlignment to only one.
            T_wc = frame["pose"] @ inverse(calibration["T_color_depth"])
            metadata["frames"].append({
                "name": name, "T_wc": T_wc.tolist(), "source_frame": frame["index"],
                "timestamp_color_us": frame["timestamps"][0],
                "timestamp_depth_us": frame["timestamps"][1],
            })
            split = "test" if order % args.test_every == args.test_every - 1 else "train"
            metadata[split].append(name)
            print(f"Exported {name} ({split})", flush=True)
    if not metadata["train"] or not metadata["test"]:
        raise ValueError("Frame selection produced an empty split; use smaller --test-every")
    shutil.copy2(mesh, target / "mesh.ply")
    (target / "cameras.json").write_text(json.dumps(metadata, indent=2) + "\n")
    provenance = {
        "scene": scene_name, "source_sens": str(recording.resolve()),
        "source_mesh": str(mesh.resolve()), "arguments": vars(args),
        "calibration": {key: value.tolist() if isinstance(value, np.ndarray) else value
                        for key, value in calibration.items()},
        "selection": "Uniform indices over valid poses, in recording order",
        "selected_frame_ids": [frame["index"] for frame in selected],
        "train_count": len(metadata["train"]), "held_out_count": len(metadata["test"]),
        "coordinate_frame": "Unaligned ScanNet mesh/world frame; OpenCV RGB camera axes",
        "depth_registration": "Depth-to-color projection, nearest-Z buffer, missing pixels zero",
        "ground_truth_usage": "Mesh copied for evaluation only; initialize fitting from training depth",
    }
    (args.out / "scene_preparation.json").write_text(json.dumps(provenance, indent=2, default=str) + "\n")
    print(f"Prepared {target}: {len(metadata['train'])} train / {len(metadata['test'])} held-out views")


if __name__ == "__main__":
    main()
