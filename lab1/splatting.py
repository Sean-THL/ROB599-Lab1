"""Lab 1 Part C -- Per-scene optimization (3D Gaussian Splatting).  [20 pts]

You are not asked to write a rasterizer. Use an existing 3DGS implementation and
record its version and commit -- C1's numbers are not reproducible without it.

C3 is the part that carries the lab's argument: report accuracy and completeness
against ground truth AT THE LAB'S FIXED THRESHOLD (``--recon-threshold``, the same
one D3 uses for the feed-forward reconstruction) so the numbers are comparable, then comment on the relationship between your PSNR and your
geometric accuracy. "They are not the same ordering, and the gap is the
finding."
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
    background = torch.zeros((1, 3), device=device)

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
            backgrounds=background,
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
    """
    raise NotImplementedError("Lab 1 C2: extract a surface from the splat model")


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
    """
    raise NotImplementedError("Lab 1 C3: evaluate the extracted geometry")
