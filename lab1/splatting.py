"""Lab 1 Part C -- Per-scene optimization (3D Gaussian Splatting).  [20 pts]

You are not asked to write a rasterizer. Use an existing 3DGS implementation and
record its version and commit -- C1's numbers are not reproducible without it.

C3 is the part that carries the lab's argument: report accuracy and completeness
against ground truth AT THE LAB'S FIXED THRESHOLD (``--recon-threshold``, the same
one D3 uses for the feed-forward reconstruction) so the numbers are comparable, then comment on the relationship between your PSNR and your
geometric accuracy. "They are not the same ordering, and the gap is the
finding."

Run a short C1 fit from the repository root:
    python -u -m lab1.splatting --data /path/to/prepared_scene --iterations 100
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class SplatModel:
    """A fitted 3DGS model, in the course conventions.

    Attributes:
        means: (N, 3) Gaussian centres, world frame, metres.
        scales: (N, 3) per-axis standard deviations in METRES (linear, > 0).
        rotations: (N, 4) unit quaternions ``(qx, qy, qz, qw)``, scalar LAST,
            mapping the Gaussian's local axes into the world frame.
        opacities: (N,) in [0, 1] (after the sigmoid).
        sh_coeffs: (N, K, 3) spherical-harmonic colour coefficients.

    Rasterizers store something else: gsplat and the reference 3DGS code keep
    LOG scales, LOGIT opacities, and SCALAR-FIRST ``(w, x, y, z)`` quaternions.
    Convert when you build this object -- ``exp``, ``sigmoid``, and a roll of
    the quaternion -- because every provided function, including
    :func:`filter_floaters`, reads these fields in the units above.
    :meth:`validate` catches the common slip of passing raw log-scales.
    """

    means: np.ndarray
    scales: np.ndarray
    rotations: np.ndarray
    opacities: np.ndarray
    sh_coeffs: np.ndarray
    fit_time_s: float = float("nan")
    peak_gpu_mb: float = float("nan")

    @property
    def n_gaussians(self) -> int:
        return len(self.means)

    def validate(self) -> None:
        """PROVIDED. Raise if a field is not in the units documented above."""
        n = len(self.means)
        shapes = {"means": (self.means, (n, 3)), "scales": (self.scales, (n, 3)),
                  "rotations": (self.rotations, (n, 4)), "opacities": (self.opacities, (n,))}
        for name, (a, want) in shapes.items():
            if np.shape(a) != want:
                raise ValueError(f"SplatModel.{name} has shape {np.shape(a)}, expected {want}")
        if np.shape(self.sh_coeffs)[:1] != (n,) or np.ndim(self.sh_coeffs) != 3:
            raise ValueError(f"SplatModel.sh_coeffs has shape {np.shape(self.sh_coeffs)}, "
                             f"expected ({n}, K, 3)")
        if n and np.min(self.scales) <= 0:
            raise ValueError("SplatModel.scales must be linear standard deviations in metres "
                             "(> 0). Negative values mean log-scales were passed: apply exp().")
        if n and (np.min(self.opacities) < 0 or np.max(self.opacities) > 1):
            raise ValueError("SplatModel.opacities must be in [0, 1]. Values outside mean "
                             "logits were passed: apply a sigmoid.")
        if n and not np.allclose(np.linalg.norm(self.rotations, axis=1), 1.0, atol=1e-3):
            raise ValueError("SplatModel.rotations must be unit quaternions (qx, qy, qz, qw).")

    def memory_mb(self) -> float:
        """PROVIDED. Resident bytes of the fitted parameters held here.

        This counts the arrays a downstream consumer must keep -- means, scales,
        rotations, opacities, SH coefficients -- at their stored dtype, which is
        the representation's cost at query time (Part E). It deliberately
        excludes training-time state (Adam moments, gradient accumulators,
        densification counters), which can be several times larger: report that
        as ``peak_gpu_mb`` from the fit, and say which number the memory column
        of your summary table holds.
        """
        return sum(np.asarray(a).nbytes for a in
                   (self.means, self.scales, self.rotations,
                    self.opacities, self.sh_coeffs)) / 1024 ** 2

    def summary(self) -> dict[str, float]:
        return {"n_gaussians": self.n_gaussians, "memory_mb": self.memory_mb(),
                "fit_time_s": self.fit_time_s, "peak_gpu_mb": self.peak_gpu_mb}


def psnr(pred: np.ndarray, gt: np.ndarray, max_val: float = 1.0) -> float:
    """PROVIDED. PSNR in dB between two images in [0, max_val]."""
    pred = np.asarray(pred, dtype=float)
    gt = np.asarray(gt, dtype=float)
    if pred.shape != gt.shape:
        raise ValueError(f"shape mismatch: {pred.shape} vs {gt.shape}")
    mse = float(np.mean((pred - gt) ** 2))
    return float("inf") if mse == 0 else float(10 * np.log10(max_val ** 2 / mse))


def filter_floaters(
    model: SplatModel, opacity_threshold: float = 0.1,
    max_scale_m: float = 0.5, k_neighbors: int = 8, max_neighbor_dist_m: float = 0.3
) -> np.ndarray:
    """PROVIDED. A defensible floater filter, returning a keep mask.

    Removes Gaussians that are (a) nearly transparent, (b) implausibly large, or
    (c) isolated from any neighbour. Provided so that everyone starts from the
    same baseline -- but C2 requires you to DOCUMENT THE PROCEDURE PRECISELY, so
    if you tune these thresholds, report the values you used and what changed.

    A splat fit always produces floaters. They barely affect PSNR, because they
    are nearly invisible from the training views, and they wreck geometric
    accuracy and collision checking. That asymmetry is the mechanism behind the
    ranking flip Part E3 asks you to find -- so do not silently tune them away
    and then report that no flip exists.
    """
    from scipy.spatial import cKDTree

    model.validate()          # linear scales and [0, 1] opacities, or this is meaningless
    keep = model.opacities > opacity_threshold
    keep &= np.max(model.scales, axis=1) < max_scale_m
    if keep.sum() > k_neighbors:
        pts = model.means[keep]
        d, _ = cKDTree(pts).query(pts, k=min(k_neighbors + 1, len(pts)))
        isolated = d[:, 1:].mean(axis=1) > max_neighbor_dist_m
        idx = np.where(keep)[0]
        keep[idx[isolated]] = False
    return keep


# --------------------------------------------------------------------------- #
# YOU WRITE THESE
# --------------------------------------------------------------------------- #
def fit_splats(images, poses, intrinsics, n_iterations: int = 30_000, **kwargs) -> SplatModel:
    """Fit 3DGS to the evaluation scene.  [C1, 8 pts]

    `images` (V, H, W, 3) in [0, 1], `poses` (V, 4, 4) camera-to-world with
    OpenCV axes (``dataset.EvalScene`` provides both), `intrinsics` (3, 3). Pass the
    TRAINING views only. gsplat's ``rasterization`` wants world-to-camera
    ``viewmats``: invert the poses with ``rob498.se3.inverse``.

    C1 wants PSNR, SSIM, and LPIPS on the HELD-OUT views, the final Gaussian
    count, wall-clock fit time, and peak GPU memory. Held-out means held out --
    if the fit ever saw those views, every number in your report is optimistic
    and the comparison against the other two representations is void.

    Required kwargs: initial_points (N, 3), world-frame metres, and
    initial_colors (N, 3), RGB floats in [0, 1], from training data only.
    Optional kwargs: device="cuda", lr=0.001, seed=0, log_every=100.

    Simple baseline: fixed Gaussian count, degree-0 SH, black background,
    and an L1 image loss. No splitting, pruning, or camera optimization.
    Fit time includes initialization, image transfers, and optimization;
    first-use CUDA kernel compilation, if needed, is also included.
    Peak memory is PyTorch-allocated memory, including all live device tensors.
    Held-out rendering and image metrics are performed separately after fitting.
    """
    import numpy as np
    import torch
    import time
    from scipy.spatial import cKDTree
    from gsplat import rasterization
    from rob498.se3 import inverse

    initial_points = kwargs.pop("initial_points", None)
    initial_colors = kwargs.pop("initial_colors", None)
    device = torch.device(kwargs.pop("device", "cuda"))
    lr = kwargs.pop("lr", 0.001)
    seed = kwargs.pop("seed", 0)
    log_every = kwargs.pop("log_every", 100)
    if kwargs:
        raise TypeError(f"Unknown fitting options: {sorted(kwargs)}")
    if initial_points is None or initial_colors is None:
        raise ValueError("Provide initial_points and initial_colors from training data")
    if device.type != "cuda" or not torch.cuda.is_available():
        raise ValueError("fit_splats requires an available CUDA GPU")
    if not isinstance(n_iterations, int) or n_iterations < 1:
        raise ValueError("n_iterations must be a positive integer")
    if not np.isfinite(lr) or lr <= 0:
        raise ValueError("lr must be positive and finite")
    if not isinstance(log_every, int) or log_every < 0:
        raise ValueError("log_every must be a nonnegative integer; 0 disables logging")

    initial_points = np.asarray(initial_points, dtype=np.float32)
    initial_colors = np.asarray(initial_colors, dtype=np.float32)
    images = np.asarray(images, dtype=np.float32)
    poses = np.asarray(poses, dtype=np.float32)
    intrinsics = np.asarray(intrinsics, dtype=np.float32)
    if (initial_points.ndim != 2 or initial_points.shape[1] != 3
            or len(initial_points) < 2):
        raise ValueError("initial_points must have shape (N, 3), with N >= 2")
    if initial_colors.shape != initial_points.shape:
        raise ValueError("initial_colors must have the same (N, 3) shape as points")
    if (images.ndim != 4 or images.shape[-1] != 3
            or min(images.shape[:3]) < 1):
        raise ValueError("images must have shape (V, H, W, 3), with V, H, W > 0")
    if poses.shape != (len(images), 4, 4) or intrinsics.shape != (3, 3):
        raise ValueError("Need one (4, 4) pose per image and (3, 3) intrinsics")
    for name, array in (("points", initial_points), ("colors", initial_colors),
                        ("images", images), ("poses", poses), ("intrinsics", intrinsics)):
        if not np.isfinite(array).all():
            raise ValueError(f"{name} contains non-finite values")
    for name, array in (("colors", initial_colors), ("images", images)):
        if array.min() < 0 or array.max() > 1:
            raise ValueError(f"{name} must contain RGB floats in [0, 1]")

    n = len(initial_points)
    rng = np.random.default_rng(seed)
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter()

    # 1. Position: put one Gaussian at each initial point.
    means = torch.nn.Parameter(
        torch.as_tensor(initial_points, dtype=torch.float32, device=device).clone()
    )

    # 2. Scale: estimate each point's spacing from its nearest neighbor.
    # The first returned neighbor is the point itself; the second is another point.
    distance, _ = cKDTree(initial_points).query(initial_points, k=2)
    spacing = np.maximum(distance[:, 1], 1e-4)

    # Start with equal sizes along x, y, and z.
    scales = torch.tensor(spacing, dtype=torch.float32, device=device)
    scales = scales[:, None].repeat(1, 3)
    log_scales = torch.nn.Parameter(torch.log(scales))

    #3. Rotation: identity quaternion in gsplat's scalar-first convention (qw, qx, qy, qz).
    rotations = torch.zeros((n, 4), device=device)
    rotations[:, 0] = 1.0
    quaternions = torch.nn.Parameter(rotations)

    # 4. Opacity: start every Gaussian at opacity 0.1.
    initial_opacity = torch.full((n,), 0.1, device=device)
    opacity_logits = torch.nn.Parameter(torch.logit(initial_opacity))

    # 5. Appearance: start with degree-0 spherical harmonics.
    # This gives each Gaussian a color that does not depend on viewing direction.
    rgb = torch.tensor(initial_colors, dtype=torch.float32, device=device)
    C0 = 0.28209479177387814
    sh_coeffs = torch.nn.Parameter(
        ((rgb - 0.5) / C0).unsqueeze(1)
    )

    # 6. Cameras: convert camera-to-world poses to world-to-camera matrices.
    viewmats = torch.tensor(
        np.stack([inverse(pose) for pose in poses]),
        dtype=torch.float32, device=device,
    )
    K = torch.tensor(intrinsics, dtype=torch.float32, device=device)
    height, width = images.shape[1:3]

    # 7. One optimizer updates all five sets of Gaussian parameters.
    # A shared learning rate is a simple starting point, not a tuned 3DGS recipe.
    optimizer = torch.optim.Adam(
        [means, log_scales, quaternions, opacity_logits, sh_coeffs], lr=lr
    )

    for step in range(n_iterations):
        # Pick a TRAINING view. Keep the image collection on CPU to save VRAM.
        view_index = int(rng.integers(len(images)))
        target = torch.tensor(images[view_index], dtype=torch.float32, device=device)
        optimizer.zero_grad(set_to_none=True)

        # Convert unrestricted parameters to valid scales, rotations and opacity.
        rendered, alpha, info = rasterization(
            means=means,
            quats=torch.nn.functional.normalize(quaternions, dim=-1),
            scales=torch.exp(log_scales),
            opacities=torch.sigmoid(opacity_logits),
            colors=sh_coeffs,
            viewmats=viewmats[view_index:view_index + 1],
            Ks=K.unsqueeze(0),
            width=width,
            height=height,
            sh_degree=0,
            # None uses black and avoids the packed backend's background-shape check.
            backgrounds=None,
            render_mode="RGB",
        )
        prediction = rendered[0]  # Remove the camera dimension: (H, W, 3).
        loss = torch.nn.functional.l1_loss(prediction, target)
        if not torch.isfinite(loss).item():
            raise RuntimeError(f"Non-finite image loss at step {step + 1}")
        loss.backward()
        optimizer.step()

        if log_every and ((step + 1) % log_every == 0 or step == 0):
            print(f"Step {step + 1}/{n_iterations} | L1 loss {loss.item():.5f}",
                  flush=True)
        # Release this view's rendering buffers before the next iteration.
        del target, rendered, alpha, info, prediction, loss

    torch.cuda.synchronize(device)
    fit_time_s = time.perf_counter() - start
    peak_gpu_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)

    # 8. Export physical values in the course conventions, not training variables.
    with torch.no_grad():
        rotations = torch.nn.functional.normalize(quaternions, dim=-1)
        model = SplatModel(
            means=means.detach().cpu().numpy().copy(),
            scales=torch.exp(log_scales).cpu().numpy(),
            rotations=rotations[:, [1, 2, 3, 0]].cpu().numpy(),  # wxyz -> xyzw
            opacities=torch.sigmoid(opacity_logits).cpu().numpy(),
            sh_coeffs=sh_coeffs.detach().cpu().numpy().copy(),
            fit_time_s=fit_time_s,
            peak_gpu_mb=peak_gpu_mb,
        )
    model.validate()
    return model


def extract_surface(model: SplatModel, method: str = "opacity_threshold", **kwargs):
    """Extract a surface or occupancy estimate from the splats.  [C2, 6 pts]

    Returns (vertices, faces) or a point set -- state which and why.

    C2 requires the procedure documented precisely: the opacity threshold, how
    you treat ANISOTROPIC Gaussians, and how you handle floaters. The
    anisotropic question is the substantive one: a Gaussian that is long and
    thin is a legitimate surface element viewed edge-on, or an artefact
    stretched along a viewing ray, and distinguishing them is the whole problem.
    Whatever rule you adopt, state it as a predicate on the scales and show what
    it removes.

    Simple baseline: return the filtered Gaussian centres as an (M, 3) point
    cloud. This is an approximate geometry estimate, not a triangle mesh or
    a Gaussian isosurface. Rotation and spatial extent are not reconstructed.

    kwargs are passed to the provided filter_floaters(): opacity_threshold=0.1,
    max_scale_m=0.5, k_neighbors=8, max_neighbor_dist_m=0.3. A Gaussian is
    rejected if ANY axis scale reaches max_scale_m, regardless of orientation;
    anisotropy alone is not rejected, so thin real structures can survive.
    """
    if method != "opacity_threshold":
        raise ValueError("Only method='opacity_threshold' is implemented")

    # The provided filter validates the model and removes low-opacity,
    # oversized and isolated Gaussians using the thresholds above.
    keep = filter_floaters(model, **kwargs)

    # One representative point per retained Gaussian. Shape: (M, 3).
    # Centres ignore the ellipsoid's shape; document this approximation in C2.
    return np.asarray(model.means)[keep].copy()


def evaluate_geometry(surface_points: np.ndarray, gt_mesh_points: np.ndarray,
                      threshold: float = 0.05):
    """Accuracy and completeness against ground truth.  [C3, 6 pts]

    Call ``rob498.metrics.accuracy_completeness`` with the lab's fixed threshold
    (``--recon-threshold``), the same one D3 uses -- the point of holding it
    fixed is that the splat and feed-forward numbers become comparable -- and
    return its ``ReconResult``.

    Then answer the actual question: how does your PSNR ordering relate to your
    geometric accuracy ordering? If the model with the best PSNR does not have
    the best accuracy, say why. The mechanism is usually that view synthesis is
    rewarded for reproducing appearance from the training viewpoints and is
    indifferent to where the geometry actually sits along the ray.

    Both point sets must be in the same world frame and in metres. Pass
    area-weighted GT mesh samples, e.g. scene.gt_points(n=200_000, seed=42),
    rather than raw mesh vertices. Reuse the same samples and threshold in D3.
    """
    from rob498.metrics import accuracy_completeness

    if not np.isfinite(threshold) or threshold <= 0:
        raise ValueError("threshold must be positive and finite, in metres")

    # Keep the provided metric definitions: prediction -> GT measures accuracy;
    # GT -> prediction measures completeness. Do not swap the two inputs.
    return accuracy_completeness(surface_points, gt_mesh_points, threshold=threshold)


def load_splats(path) -> SplatModel:
    """Read parameters saved by the fitting command; never load pickled objects."""
    with np.load(path, allow_pickle=False) as saved:
        model = SplatModel(
            means=saved["means"], scales=saved["scales"],
            rotations=saved["rotations"], opacities=saved["opacities"],
            sh_coeffs=saved["sh_coeffs"],
            fit_time_s=float(saved["fit_time_s"]),
            peak_gpu_mb=float(saved["peak_gpu_mb"]),
        )
    model.validate()
    return model


def evaluate_rendering(model: SplatModel, scene, out_dir):
    """C1: render ONLY held-out cameras and average full-image RGB metrics.

    Scores use clipped [0, 1] float images before PNG quantization. SSIM uses
    an 11-pixel Gaussian window (sigma=1.5), population covariance, data_range=1.
    LPIPS uses AlexNet, metric version 0.1, with NCHW RGB normalized to [-1, 1].
    The first LPIPS invocation may download pretrained metric-network weights.
    """
    import csv
    from importlib.metadata import version
    from pathlib import Path
    import torch
    import lpips
    from PIL import Image, ImageDraw
    from gsplat import rasterization
    from skimage.metrics import structural_similarity
    from rob498.io3d import write_json
    from rob498.se3 import inverse

    model.validate()
    if not scene.test or set(scene.train) & set(scene.test):
        raise ValueError("Need nonempty held-out views disjoint from training views")
    if min(scene.width, scene.height) < 11:
        raise ValueError("Images must be at least 11 pixels per axis for this SSIM setup")
    if not torch.cuda.is_available():
        raise RuntimeError("Held-out Gaussian rendering requires a CUDA GPU")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=False)
    for folder in ("rendered", "reference", "comparisons"):
        (out / folder).mkdir()

    # Saved SplatModel uses physical scales and xyzw; gsplat wants wxyz.
    means = torch.tensor(model.means, dtype=torch.float32, device="cuda")
    scales = torch.tensor(model.scales, dtype=torch.float32, device="cuda")
    quats = torch.tensor(model.rotations[:, [3, 0, 1, 2]], dtype=torch.float32, device="cuda")
    opacities = torch.tensor(model.opacities, dtype=torch.float32, device="cuda")
    colors = torch.tensor(model.sh_coeffs, dtype=torch.float32, device="cuda")
    K = torch.tensor(scene.K, dtype=torch.float32, device="cuda").unsqueeze(0)
    sh_degree = int(round(np.sqrt(model.sh_coeffs.shape[1]))) - 1
    if (sh_degree + 1) ** 2 != model.sh_coeffs.shape[1]:
        raise ValueError("SH coefficient count must be a square: 1, 4, 9, ...")
    perceptual = lpips.LPIPS(net="alex", version="0.1").cuda().eval()
    rows = []

    with torch.no_grad():
        for name in scene.test:
            reference = scene.image(name)
            if reference.shape != (scene.height, scene.width, 3):
                raise ValueError(f"{name}: image dimensions disagree with cameras.json")
            viewmat = torch.tensor(inverse(scene.pose(name)), dtype=torch.float32,
                                   device="cuda").unsqueeze(0)
            rendered, alpha, info = rasterization(
                means=means, quats=quats, scales=scales, opacities=opacities,
                colors=colors, viewmats=viewmat, Ks=K,
                width=scene.width, height=scene.height, sh_degree=sh_degree,
                backgrounds=None, render_mode="RGB",
            )
            prediction_tensor = rendered[0].clamp(0, 1)
            prediction = prediction_tensor.cpu().numpy()
            target_tensor = torch.tensor(reference, dtype=torch.float32, device="cuda")
            prediction_nchw = prediction_tensor.permute(2, 0, 1).unsqueeze(0)
            target_nchw = target_tensor.permute(2, 0, 1).unsqueeze(0)
            row = {
                "view": name,
                "psnr_db": psnr(prediction, reference),
                "ssim": float(structural_similarity(
                    reference, prediction, channel_axis=-1, data_range=1.0,
                    gaussian_weights=True, sigma=1.5, use_sample_covariance=False,
                )),
                "lpips": float(perceptual(prediction_nchw * 2 - 1, target_nchw * 2 - 1).item()),
                "mean_opacity": float(alpha.mean().item()),
            }
            rows.append(row)
            print(f"{name} | PSNR {row['psnr_db']:.2f} dB | "
                  f"SSIM {row['ssim']:.4f} | LPIPS {row['lpips']:.4f}", flush=True)

            # Save the exact same view side by side, without a display server.
            reference_image = Image.fromarray(np.rint(reference * 255).astype(np.uint8))
            prediction_image = Image.fromarray(np.rint(prediction * 255).astype(np.uint8))
            reference_image.save(out / "reference" / f"{name}.png")
            prediction_image.save(out / "rendered" / f"{name}.png")
            pair = Image.new("RGB", (2 * scene.width, scene.height + 28), "white")
            pair.paste(reference_image, (0, 28))
            pair.paste(prediction_image, (scene.width, 28))
            draw = ImageDraw.Draw(pair)
            draw.text((8, 7), "Real held-out image", fill="black")
            draw.text((scene.width + 8, 7), "Gaussian rendering", fill="black")
            pair.save(out / "comparisons" / f"{name}.png")
            del rendered, alpha, info, prediction_tensor, target_tensor
            del prediction_nchw, target_nchw

    with (out / "per_view_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        **model.summary(),
        "psnr_mean_db": float(np.mean([row["psnr_db"] for row in rows])),
        "ssim_mean": float(np.mean([row["ssim"] for row in rows])),
        "lpips_mean": float(np.mean([row["lpips"] for row in rows])),
        "held_out_views": scene.test, "n_held_out_views": len(rows),
        "resolution": [scene.width, scene.height], "sh_degree": sh_degree,
        "evaluation_scope": "all RGB pixels; floats clipped to [0,1]; no opacity mask",
        "aggregation": "arithmetic mean of per-view metrics",
        "ssim_settings": "Gaussian window 11, sigma 1.5, population covariance, data_range 1",
        "lpips_settings": "AlexNet, version 0.1, RGB NCHW normalized to [-1,1]",
        "gsplat_version": version("gsplat"), "lpips_package_version": version("lpips"),
        "skimage_version": version("scikit-image"),
    }
    write_json(out / "metrics.json", summary)
    return summary


def save_geometry_results(model: SplatModel, scene, out_dir, *, threshold=0.05,
                          n_gt_points=200000, seed=42, opacity_threshold=0.1,
                          max_scale_m=0.5, k_neighbors=8, max_neighbor_dist_m=0.3):
    """C2/C3: save extracted points, GT samples, errors, metrics and an overview."""
    from pathlib import Path
    from scipy.spatial import cKDTree
    from rob498.io3d import write_json, write_ply_points
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=False)
    filters = {
        "opacity_threshold": opacity_threshold, "max_scale_m": max_scale_m,
        "k_neighbors": k_neighbors, "max_neighbor_dist_m": max_neighbor_dist_m,
    }
    surface = extract_surface(model, **filters)
    write_ply_points(out / "surface_points.ply", surface)
    gt = scene.gt_points(n=n_gt_points, seed=seed)
    np.savez_compressed(out / "gt_samples.npz", points=gt, seed=seed)
    write_ply_points(out / "gt_points.ply", gt)
    summary = {
        "method": "filtered Gaussian centres; shape and orientation not reconstructed",
        "filters": filters, "gaussians_before": model.n_gaussians,
        "points_after": len(surface), "gt_sample_count": n_gt_points,
        "gt_sampling_seed": seed, "threshold_m": threshold,
    }
    write_json(out / "extraction.json", summary)
    if not len(surface):
        raise ValueError("All Gaussians were filtered out. Empty extraction was saved; do not invent geometry scores")
    geometry = evaluate_geometry(surface, gt, threshold=threshold)
    summary.update(geometry.summary())
    np.savez_compressed(
        out / "distances.npz", dist_pred_to_gt=geometry.dist_pred_to_gt,
        dist_gt_to_pred=geometry.dist_gt_to_pred,
    )

    # An alignment diagnostic from TRAINING sensor depth, not mesh-rendered depth.
    sensor_points, _ = scene.fused_points(names=scene.train, stride=4, voxel=0.05)
    if len(sensor_points):
        distances, _ = cKDTree(gt).query(sensor_points)
        summary["training_sensor_to_gt_median_m"] = float(np.median(distances))
    write_json(out / "metrics.json", summary)

    # Compare geometry in equal-scale orthographic projections, no GUI needed.
    rng = np.random.default_rng(seed)
    visible_gt = gt[rng.choice(len(gt), min(len(gt), 10000), replace=False)]
    visible_surface = surface[rng.choice(len(surface), min(len(surface), 10000), replace=False)]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for axis, (a, b) in zip(axes, ((0, 1), (0, 2), (1, 2))):
        axis.scatter(visible_gt[:, a], visible_gt[:, b], s=1, c="gray", alpha=0.3,
                     label="GT mesh samples", rasterized=True)
        axis.scatter(visible_surface[:, a], visible_surface[:, b], s=2, c="red", alpha=0.5,
                     label="Extracted centres", rasterized=True)
        axis.set_xlabel("xyz"[a] + " (m)")
        axis.set_ylabel("xyz"[b] + " (m)")
        axis.set_aspect("equal", adjustable="box")
        axis.legend(markerscale=3)
    fig.tight_layout()
    fig.savefig(out / "surface_vs_gt.png", dpi=150)
    plt.close(fig)
    return summary


def main():
    """Fit or evaluate a saved model without a pasted Python block."""
    import argparse
    import json
    from datetime import datetime
    from importlib.metadata import version
    from pathlib import Path
    import sys

    # Also support `python lab1/splatting.py ...` from the repository checkout.
    if not __package__:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from lab1.dataset import load_eval_scene
    from rob498.config import RunManifest

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True,
                        help="Prepared data root containing eval_scene/cameras.json")
    parser.add_argument("--out", type=Path,
                        help="New output folder; default: results/c1_<timestamp>")
    parser.add_argument("--load", type=Path, help="Saved splats.npz; skip fitting")
    parser.add_argument("--evaluate", action="store_true",
                        help="Render held-out views, save image pairs and PSNR/SSIM/LPIPS")
    parser.add_argument("--geometry", action="store_true",
                        help="Run C2/C3, save extracted geometry, metrics and GT comparison")
    parser.add_argument("--recon-threshold", type=float, default=0.05)
    parser.add_argument("--gt-points", type=int, default=200000)
    parser.add_argument("--opacity-threshold", type=float, default=0.1)
    parser.add_argument("--max-scale-m", type=float, default=0.5)
    parser.add_argument("--k-neighbors", type=int, default=8)
    parser.add_argument("--max-neighbor-dist-m", type=float, default=0.3)
    parser.add_argument("--iterations", type=int, default=100,
                        help="100 for a short first fit; choose a larger budget for final fitting")
    parser.add_argument("--max-gaussians", type=int, default=20000)
    parser.add_argument("--voxel-size", type=float, default=0.05,
                        help="Training-depth cloud downsampling size in metres")
    parser.add_argument("--stride", type=int, default=4,
                        help="Pixel stride when back-projecting training depth")
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-every", type=int, default=10)
    args = parser.parse_args()
    if args.iterations < 1 or args.max_gaussians < 2 or args.stride < 1:
        parser.error("Need iterations >= 1, max-gaussians >= 2 and stride >= 1")
    if not np.isfinite(args.voxel_size) or args.voxel_size <= 0:
        parser.error("voxel-size must be positive and finite")
    if not np.isfinite(args.lr) or args.lr <= 0 or args.log_every < 0:
        parser.error("Need positive finite lr and nonnegative log-every")
    if args.load is not None and not (args.evaluate or args.geometry):
        parser.error("With --load, select --evaluate and/or --geometry")
    if args.gt_points < 1 or args.k_neighbors < 1:
        parser.error("gt-points and k-neighbors must be positive")
    if not np.isfinite(args.opacity_threshold) or not 0 <= args.opacity_threshold <= 1:
        parser.error("opacity-threshold must be in [0, 1]")
    for value in (args.recon_threshold, args.max_scale_m, args.max_neighbor_dist_m):
        if not np.isfinite(value) or value <= 0:
            parser.error("Geometry distance/scale thresholds must be positive and finite")

    # 1. Load the scene. Existing exported data is reused, not regenerated.
    scene = load_eval_scene(args.data)
    if not scene.train:
        parser.error("The scene has no training views")
    print(f"Scene: {args.data} | {len(scene.train)} train / {len(scene.test)} held-out views",
          flush=True)

    # 3. Record the exact inputs and settings alongside the fitted model.
    prefix = "c_evaluation" if args.load else "c1"
    out = args.out or Path("results") / (prefix + datetime.now().strftime("_%Y%m%d_%H%M%S_%f"))
    out.mkdir(parents=True, exist_ok=False)
    manifest = RunManifest(run_name=prefix, seed=args.seed)
    manifest.add_param("data_root", str(args.data.resolve()))
    manifest.add_param("training_views", scene.train)
    manifest.add_param("held_out_views", scene.test)
    manifest.add_param("resolution", [scene.width, scene.height])
    manifest.add_param("gsplat_version", version("gsplat"))
    if args.load is not None:
        model = load_splats(args.load)
        manifest.add_param("source_model", str(args.load.resolve()))
        source_manifest = args.load.parent / "manifest.json"
        if source_manifest.exists():
            source = json.loads(source_manifest.read_text())
            recorded = source["params"]
            if (recorded.get("training_views") != scene.train
                    or recorded.get("held_out_views") != scene.test
                    or recorded.get("resolution") != [scene.width, scene.height]
                    or recorded.get("data_root") != str(args.data.resolve())):
                parser.error("Saved model scene/split/resolution differs from the requested scene")
            manifest.add_param("source_fit_manifest", source)
        else:
            print("No source manifest: scene/split provenance must be checked manually.", flush=True)
        print(f"Loaded {model.n_gaussians} Gaussians; no fitting will run", flush=True)
    else:
        # 2. Build initial Gaussians from TRAINING sensor depth, never the GT mesh.
        points, colors = scene.fused_points(
            names=scene.train, stride=args.stride, voxel=args.voxel_size
        )
        if len(points) < 2:
            parser.error("Training depth produced fewer than two valid points")
        if len(points) > args.max_gaussians:
            indices = np.random.default_rng(args.seed).choice(
                len(points), args.max_gaussians, replace=False
            )
            points, colors = points[indices], colors[indices]
        print(f"Initial Gaussians: {len(points)}", flush=True)
        for key, value in {
            "iterations": args.iterations, "initial_gaussian_count": len(points),
            "initialization_stride": args.stride, "initialization_voxel_size_m": args.voxel_size,
            "lr": args.lr, "sh_degree": 0,
            "gaussian_count_strategy": "fixed; no splitting or pruning",
            "loss": "L1 RGB on training views; black background",
        }.items():
            manifest.add_param(key, value)
        manifest.save(out)

        # 4. Fit, then save NumPy parameters in the course's SplatModel conventions.
        model = fit_splats(
            scene.images(scene.train), scene.poses(scene.train), scene.K,
            initial_points=points,
            initial_colors=colors.astype(np.float32) / 255.0,
            n_iterations=args.iterations, lr=args.lr, seed=args.seed,
            log_every=args.log_every,
        )
        np.savez_compressed(out / "splats.npz", **vars(model))
    manifest.add_result("fit", model.summary())
    manifest.save(out)
    print(model.summary(), flush=True)

    if args.evaluate:
        rendering = evaluate_rendering(model, scene, out / "rendering")
        manifest.add_result("c1_rendering", rendering)
        manifest.save(out)
        print(f"Held-out means: PSNR={rendering['psnr_mean_db']:.2f} dB, "
              f"SSIM={rendering['ssim_mean']:.4f}, LPIPS={rendering['lpips_mean']:.4f}", flush=True)
    if args.geometry:
        geometry = save_geometry_results(
            model, scene, out / "geometry", threshold=args.recon_threshold,
            n_gt_points=args.gt_points, seed=args.seed,
            opacity_threshold=args.opacity_threshold, max_scale_m=args.max_scale_m,
            k_neighbors=args.k_neighbors, max_neighbor_dist_m=args.max_neighbor_dist_m,
        )
        manifest.add_result("c2_c3_geometry", geometry)
        manifest.save(out)
        print(f"Geometry: {geometry}", flush=True)
    print(f"Saved outputs and conditions to {out.resolve()}", flush=True)
    if not args.evaluate:
        print("Held-out rendering metrics are still required to complete C1.", flush=True)


if __name__ == "__main__":
    main()
