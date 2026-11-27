"""Lab 1 Part D -- Feed-forward reconstruction (VGGT).  [15 pts]

D1 is explicit: supply NO POSES AND NO INTRINSICS. The point of the comparison
is what a feed-forward model recovers from images alone. Feeding it poses makes
its numbers incomparable to the premise of the part.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class FeedForwardResult:
    """What the model returns, plus the cost of getting it.

    Everything here is in the MODEL'S frame and scale -- for VGGT, the first
    input camera's frame, in units that are not metres. Map it into the world
    with the Sim(3) from :func:`align_to_ground_truth` before comparing it with
    anything metric, and say in the report that you did.
    """

    points: np.ndarray                 # (N, 3) predicted geometry, model frame
    poses: np.ndarray                  # (V, 4, 4) predicted camera-to-world, OpenCV axes
    intrinsics: np.ndarray | None      # (V, 3, 3) if predicted
    n_views: int
    wall_clock_s: float
    peak_gpu_mb: float
    confidence: np.ndarray | None = None   # (N,) if the model provides it
    view_ids: list | None = None       # which input frames, in order -- D2 needs
                                       # them to pair predicted poses with GT poses

    def summary(self) -> dict[str, float]:
        return {"n_views": self.n_views, "n_points": len(self.points),
                "wall_clock_s": self.wall_clock_s, "peak_gpu_mb": self.peak_gpu_mb}

    def memory_mb(self) -> float:
        """PROVIDED. Resident bytes of the reconstruction a consumer keeps."""
        arrays = [self.points, self.poses]
        arrays += [a for a in (self.intrinsics, self.confidence) if a is not None]
        return sum(np.asarray(a).nbytes for a in arrays) / 1024 ** 2


def apply_sim3(S: np.ndarray, points: np.ndarray) -> np.ndarray:
    """PROVIDED. Apply a 4x4 similarity ``[[s R, t], [0, 1]]`` to (N, 3) points."""
    S = np.asarray(S, dtype=float)
    return np.asarray(points, dtype=float) @ S[:3, :3].T + S[:3, 3]


def align_to_ground_truth(pred_poses: np.ndarray, gt_poses: np.ndarray) -> dict:
    """PROVIDED. Sim(3)-align predicted poses to ground truth.  [D2, 5 pts]

    Both arguments are (V, 4, 4) camera-to-world poses for the SAME views in the
    same order. VGGT returns world-to-camera extrinsics (OpenCV "camera from
    world"); invert them before calling this.

    Returns ATE, rotation error in degrees, the recovered SCALE FACTOR, and
    ``sim3``: the (4, 4) similarity mapping the model's frame onto the world.
    Use the same ``sim3`` for the D3 geometry and the Part E queries, so that all
    three are evaluated under one alignment.

    The alignment uses camera centres only (Umeyama), so it needs at least three
    views that are not collinear; a camera path along a straight line leaves the
    rotation about that line undetermined, and the numbers will look worse than
    the reconstruction is.

    D2 asks you to note the scale factor and what it means for a robot that
    needs metric distances. The honest reading: a scale recovered by aligning to
    ground truth is not available to a robot that has no ground truth, so a
    method whose output is only correct up to an unknown scale has not solved
    the metric-reconstruction problem no matter how good its aligned ATE looks.
    Say that in your own words, with your number attached.
    """
    from rob498.metrics import ate_rmse, umeyama_alignment
    from rob498.se3 import rotation_angle

    pred = np.asarray(pred_poses, dtype=float)
    gt = np.asarray(gt_poses, dtype=float)
    if pred.shape != gt.shape or pred.ndim != 3 or pred.shape[1:] != (4, 4):
        raise ValueError(f"need matching (V, 4, 4) pose arrays, got {pred.shape}, {gt.shape}")
    if len(pred) < 3:
        raise ValueError("Sim(3) alignment needs at least 3 views")
    S, scale = umeyama_alignment(pred[:, :3, 3], gt[:, :3, 3], with_scale=True)
    R_align = S[:3, :3] / scale
    ate = ate_rmse(pred, gt, align=True, with_scale=True)
    # Rotation error AFTER alignment: the predicted rotations live in the model's
    # frame, so comparing them to GT without R_align measures the frame offset.
    rot = [np.degrees(rotation_angle((R_align @ p[:3, :3]).T @ g[:3, :3]))
           for p, g in zip(pred, gt)]
    return {
        "ate_rmse_m": ate["rmse"],
        "ate_median_m": ate["median"],
        "scale_factor": float(scale),
        "mean_rotation_error_deg": float(np.mean(rot)),
        "median_rotation_error_deg": float(np.median(rot)),
        "n_views": len(pred),
        "sim3": S,
    }


# --------------------------------------------------------------------------- #
# YOU WRITE THESE
# --------------------------------------------------------------------------- #
def run_vggt(images, n_views: int | None = None, **kwargs) -> FeedForwardResult:
    """Run VGGT on the images, with no poses and no intrinsics.  [D1, 5 pts]

    ``images`` is a list of RGB file paths. Optional keyword settings are
    ``model`` (reuse loaded VGGT weights), ``view_ids`` (ordered frame names),
    and ``confidence_threshold`` (depth-head score, default 0; not probability).
    This implementation uses predicted depth and cameras for the point cloud.

    Return a :class:`FeedForwardResult` with camera-to-world ``poses``. VGGT's
    ``pose_encoding_to_extri_intri`` returns world-to-camera extrinsics, 3x4,
    OpenCV axes ("camera from world"): pad to 4x4 and invert. Record which
    frames you used in ``view_ids``.

    D1 asks how wall-clock and peak memory SCALE as you vary the number of input
    views. Run at least four view counts and plot it. The scaling is the
    interesting quantity -- attention over views is not linear, and the point at
    which the model stops fitting in memory is a deployment fact about the
    method, not an incidental detail of your GPU.
    """
    # images is a list of RGB image FILE PATHS, not ground-truth camera data.
    # Optional settings: model (reuse loaded weights), view_ids, confidence_threshold.
    import time
    from pathlib import Path
    import torch
    from vggt.models.vggt import VGGT
    from vggt.utils.load_fn import load_and_preprocess_images
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri
    from vggt.utils.geometry import unproject_depth_map_to_point_map

    model = kwargs.pop("model", None)
    view_ids = kwargs.pop("view_ids", None)
    confidence_threshold = kwargs.pop("confidence_threshold", 0.0)
    if kwargs:
        raise TypeError(f"Unknown settings: {list(kwargs)}. Supply RGB paths only, no camera inputs.")
    paths = list(images)
    if not paths or not all(isinstance(path, (str, Path)) for path in paths):
        raise ValueError("images must be a nonempty list of RGB image file paths")
    if n_views is not None:
        if not isinstance(n_views, int) or not 1 <= n_views <= len(paths):
            raise ValueError("n_views must be between 1 and the number of images")
        paths = paths[:n_views]
    if view_ids is None:
        view_ids = [Path(path).stem for path in paths]
    else:
        view_ids = list(view_ids)[:len(paths)]
    if len(view_ids) != len(paths) or len(set(view_ids)) != len(view_ids):
        raise ValueError("Need one unique view ID per selected image, in the same order")
    if not np.isfinite(confidence_threshold) or confidence_threshold < 0:
        raise ValueError("confidence_threshold must be finite and nonnegative")
    if not torch.cuda.is_available():
        raise RuntimeError("Run VGGT in the rob498 environment on your NVIDIA GPU")

    # 1. Load RELEASED weights, not a model trained from scratch for this scene.
    # Download/model loading is excluded from the measured reconstruction time.
    if model is None:
        print("Loading facebook/VGGT-1B released weights (first use downloads them)", flush=True)
        model = VGGT.from_pretrained("facebook/VGGT-1B")
    model = model.to("cuda").eval()
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start_time = time.perf_counter()

    # 2. Use the official RGB resize/centre-crop preprocessing.
    # Predicted intrinsics describe THESE processed images, not the original size.
    image_tensor = load_and_preprocess_images([str(path) for path in paths]).to("cuda")
    # Run VGGT on the images
    with torch.inference_mode():
        with torch.autocast(device_type="cuda", dtype=dtype):
            predictions = model(image_tensor)
        # 3. Decode the cameras estimated by VGGT, not dataset cameras.
        extrinsics, intrinsics = pose_encoding_to_extri_intri(
            predictions["pose_enc"].float(), image_tensor.shape[-2:]
        )
        extrinsics = extrinsics[0].float().cpu().numpy()
        intrinsics = intrinsics[0].float().cpu().numpy()
        depth = predictions["depth"][0].float().cpu().numpy()
        confidence = predictions["depth_conf"][0].float().cpu().numpy()

    # 4. Back-project PREDICTED depth using PREDICTED cameras.
    # These coordinates have an unknown scale; they are not yet world metres.
    point_maps = unproject_depth_map_to_point_map(depth, extrinsics, intrinsics)
    valid = (np.isfinite(point_maps).all(axis=-1) & np.isfinite(depth[..., 0])
             & (depth[..., 0] > 0) & np.isfinite(confidence)
             & (confidence > confidence_threshold))
    points = point_maps[valid].astype(np.float32)
    confidence = confidence[valid].astype(np.float32)
    if not len(points):
        raise ValueError("No valid predicted points remain at this confidence threshold")

    # 5. VGGT returns world-to-camera [R|t]. Our result needs camera-to-world.
    world_to_camera = np.repeat(np.eye(4)[None], len(paths), axis=0)
    world_to_camera[:, :3, :] = extrinsics
    poses = np.linalg.inv(world_to_camera).astype(np.float32)
    if not np.isfinite(poses).all() or not np.isfinite(intrinsics).all():
        raise ValueError("VGGT produced non-finite camera predictions")
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start_time
    peak_gpu_mb = torch.cuda.max_memory_allocated() / 1024 ** 2

    return FeedForwardResult(
        points=points, poses=poses, intrinsics=intrinsics, n_views=len(paths),
        wall_clock_s=elapsed, peak_gpu_mb=peak_gpu_mb,
        confidence=confidence, view_ids=view_ids,
    )


def evaluate_geometry(result: FeedForwardResult, gt_mesh_points: np.ndarray,
                      threshold: float = 0.05, sim3: np.ndarray | None = None):
    """Accuracy and completeness against ground truth.  [D3, 5 pts]

    Same metrics and same threshold as C3 (``--recon-threshold``), so the splat
    and feed-forward numbers are comparable. `sim3` is
    ``align_to_ground_truth(...)["sim3"]``: the predicted points are in the
    model's frame and scale, and comparing them to the GT mesh unaligned
    measures the frame offset, not the geometry. Returns a
    ``rob498.metrics.ReconResult``.

    D3 asks you to IDENTIFY THE REGIONS WHERE IT FAILS AND CHARACTERIZE THEM --
    textureless, specular, thin structure, or beyond the view distribution.
    ``ReconResult.dist_pred_to_gt`` gives you per-point error; colour the cloud
    by it, look at the worst decile, and name what those regions have in common.
    A histogram alone does not answer this; you have to look at where the errors
    are.
    """
    from rob498.metrics import accuracy_completeness

    # A missing transform must not silently score arbitrary model units as metres.
    if sim3 is None:
        raise ValueError("Pass align_to_ground_truth(...)[\"sim3\"] before evaluating geometry")
    if not np.isfinite(threshold) or threshold <= 0:
        raise ValueError("threshold must be positive and finite, in metres")
    S = np.asarray(sim3, dtype=float)
    if S.shape != (4, 4) or not np.isfinite(S).all():
        raise ValueError("sim3 must be a finite (4, 4) matrix")
    if not np.allclose(S[3], [0, 0, 0, 1]):
        raise ValueError("sim3 must have last row [0, 0, 0, 1]")
    scale = np.linalg.norm(S[:3, 0])
    if scale <= 0 or not np.allclose(
            S[:3, :3].T @ S[:3, :3], scale ** 2 * np.eye(3), rtol=1e-5, atol=1e-10):
        raise ValueError("sim3 must contain one positive uniform scale and a rotation")
    if np.linalg.det(S[:3, :3]) <= 0:
        raise ValueError("sim3 rotation must not reflect the reconstruction")
    for name, points in (("prediction", result.points), ("ground truth", gt_mesh_points)):
        points = np.asarray(points)
        if points.ndim != 2 or points.shape[1:] != (3,) or not len(points):
            raise ValueError(f"{name} points must be a nonempty (N, 3) array")
        if not np.isfinite(points).all():
            raise ValueError(f"{name} points must be finite")

    # Apply the SAME camera-derived Sim(3) to every predicted point, exactly once.
    # Leave result.points in model coordinates so it can be reused safely.
    aligned_points = apply_sim3(S, result.points)
    return accuracy_completeness(aligned_points, gt_mesh_points, threshold=threshold)


def main():
    """Run D1 and optionally D2/D3, using ground truth only after inference."""
    import argparse
    from datetime import datetime
    from pathlib import Path
    from importlib.metadata import version
    from lab1.dataset import load_eval_scene
    from rob498.config import RunManifest
    from rob498.io3d import write_json, write_ply_points

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True,
                        help="Prepared data root containing eval_scene/")
    parser.add_argument("--n-views", type=int, default=4)
    parser.add_argument("--out", type=Path, help="New output folder; default is timestamped")
    parser.add_argument("--confidence-threshold", type=float, default=0.0,
                        help="Keep depth confidence above this value; not a probability")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--evaluate", action="store_true",
                        help="After inference: align predicted cameras and evaluate geometry")
    parser.add_argument("--recon-threshold", type=float, default=0.05,
                        help="Distance threshold in metres; use the same value as C3")
    parser.add_argument("--gt-points", type=int, default=200000)
    parser.add_argument("--gt-samples", type=Path,
                        help="Reuse C3 geometry/gt_samples.npz for identical GT samples")
    args = parser.parse_args()
    scene = load_eval_scene(args.data)
    if not 1 <= args.n_views <= len(scene.train):
        parser.error(f"n-views must be between 1 and {len(scene.train)}")
    if not np.isfinite(args.confidence_threshold) or args.confidence_threshold < 0:
        parser.error("confidence-threshold must be finite and nonnegative")
    if args.evaluate and args.n_views < 3:
        parser.error("D2 alignment needs at least three views")
    if args.gt_points < 1 or not np.isfinite(args.recon_threshold) or args.recon_threshold <= 0:
        parser.error("Need positive gt-points and a positive finite recon-threshold")

    # Spread selected views across the TRAINING sequence. Never use held-out RGB.
    indices = np.linspace(0, len(scene.train) - 1, args.n_views, dtype=int)
    names = [scene.train[index] for index in indices]
    paths = [scene.image_path(name) for name in names]
    out = args.out or Path("results") / datetime.now().strftime("d1_%Y%m%d_%H%M%S_%f")
    out.mkdir(parents=True, exist_ok=False)
    manifest = RunManifest(run_name="vggt_d1", seed=args.seed)
    manifest.add_param("data_root", str(args.data.resolve()))
    manifest.add_param("view_ids", names)
    manifest.add_param("model", "facebook/VGGT-1B released weights")
    manifest.add_param("vggt_version", version("vggt"))
    manifest.add_param("preprocessing", "official load_and_preprocess_images, crop mode")
    manifest.add_param("point_source", "predicted depth unprojected with predicted cameras")
    manifest.add_param("confidence_threshold", args.confidence_threshold)
    manifest.add_param("timing_scope", "RGB preprocessing, transfers, forward, decoding, point conversion; excludes model loading/download; no warmup")
    manifest.add_param("memory_scope", "peak PyTorch CUDA allocation, includes model weights; excludes reserved memory and non-PyTorch allocations")
    manifest.add_param("input_cameras", "none; scene cameras are not passed to VGGT")
    manifest.save(out)
    print(f"VGGT input: {len(names)} RGB images, no poses or intrinsics", flush=True)
    result = run_vggt(paths, view_ids=names,
                      confidence_threshold=args.confidence_threshold)
    np.savez_compressed(out / "predictions.npz", **vars(result))
    write_ply_points(out / "points_model_frame.ply", result.points)
    manifest.add_result("d1", result.summary())
    manifest.save(out)
    print(result.summary(), flush=True)
    if args.evaluate:
        # D2: pair predicted and GT cameras by frame ID, never by an assumed order.
        # These GT poses are used ONLY for evaluation, never supplied to VGGT.
        gt_poses = scene.poses(result.view_ids)
        for label, poses in (("predicted", result.poses), ("ground-truth", gt_poses)):
            if not np.isfinite(poses).all():
                raise ValueError(f"Non-finite {label} poses")
            centres = poses[:, :3, 3]
            if np.linalg.matrix_rank(centres - centres.mean(axis=0), tol=1e-6) < 2:
                raise ValueError(f"{label} camera centres are collinear; select more/different views")
        alignment = align_to_ground_truth(result.poses, gt_poses)
        S = alignment["sim3"]
        if not np.isfinite(S).all() or alignment["scale_factor"] <= 0:
            raise ValueError("Camera alignment did not produce a finite positive scale")
        write_json(out / "alignment.json", alignment)
        np.save(out / "sim3.npy", S)

        # Scale/rotate/translate camera CENTRES; camera rotations get rotation only.
        aligned_poses = result.poses.astype(float).copy()
        aligned_poses[:, :3, :3] = (S[:3, :3] / alignment["scale_factor"]) @ result.poses[:, :3, :3]
        aligned_poses[:, :3, 3] = apply_sim3(S, result.poses[:, :3, 3])
        np.savez_compressed(out / "aligned_cameras.npz", poses=aligned_poses,
                            gt_poses=gt_poses, view_ids=result.view_ids)

        # D3: reuse C3 samples, or use its default area sampling count and seed.
        if args.gt_samples:
            with np.load(args.gt_samples, allow_pickle=False) as saved:
                gt_points = saved["points"]
            manifest.add_param("gt_sample_source", str(args.gt_samples.resolve()))
        else:
            gt_points = scene.gt_points(n=args.gt_points, seed=args.seed)
            manifest.add_param("gt_sampling_seed", args.seed)
        geometry = evaluate_geometry(result, gt_points, args.recon_threshold, sim3=S)
        world_points = apply_sim3(S, result.points)
        write_ply_points(out / "points_world.ply", world_points)
        write_ply_points(out / "gt_points.ply", gt_points)
        np.savez_compressed(out / "gt_samples.npz", points=gt_points)
        np.savez_compressed(out / "geometry_distances.npz",
                            dist_pred_to_gt=geometry.dist_pred_to_gt,
                            dist_gt_to_pred=geometry.dist_gt_to_pred)
        # Blue = close to GT; red = error at least 3 times the metric threshold.
        error_fraction = np.clip(geometry.dist_pred_to_gt / (3 * args.recon_threshold), 0, 1)
        colors = np.zeros((len(world_points), 3), dtype=np.uint8)
        colors[:, 0] = np.rint(255 * error_fraction).astype(np.uint8)
        colors[:, 2] = np.rint(255 * (1 - error_fraction)).astype(np.uint8)
        write_ply_points(out / "points_by_error.ply", world_points, colors)
        write_json(out / "geometry_metrics.json", geometry.summary())
        manifest.add_param("recon_threshold_m", args.recon_threshold)
        manifest.add_param("alignment_method", "one camera-centre Umeyama Sim(3); no mesh ICP")
        manifest.add_param("gt_sample_count", len(gt_points))
        manifest.add_param("error_colors", "blue=0m, red>=3*threshold; full point cloud, no error filtering")
        manifest.add_result("d2", alignment)
        manifest.add_result("d3", geometry.summary())
        manifest.save(out)
        print(f"Alignment: scale={alignment['scale_factor']:.4f}, "
              f"ATE={alignment['ate_rmse_m']:.4f} m, "
              f"rotation={alignment['mean_rotation_error_deg']:.2f} deg", flush=True)
        print(f"Geometry: {geometry.summary()}", flush=True)
    else:
        print("Points are NOT metric-aligned: add --evaluate for D2/D3.", flush=True)
    print(f"Saved results to {out.resolve()}", flush=True)


if __name__ == "__main__":
    main()
