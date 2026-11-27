#!/usr/bin/env python3
"""Lab 1 driver.

    python lab1/run_lab1.py --data data/lab1 --out results/ --part all

`--data` is the directory fetch_data.py unpacked; lab1/dataset.py documents its
layout and loads it.

Produces results/tables.md containing the representation-comparison table that
the handout calls "the centre of the report", plus manifest.json.

Part E needs B, C, and D to have run -- it compares them. Run the earlier parts
first and cache their outputs to --out.
"""
from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rob498 import report                          # noqa: E402
from rob498.config import RunManifest              # noqa: E402

PARTS = ["b", "c", "d", "e"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("results"))
    ap.add_argument("--part", choices=[*PARTS, "all"], default="all")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--b-voxel-sizes", nargs="+", type=float, default=[0.05],
                    help="Metres; train an independent Part B model for each size")
    ap.add_argument("--b-epochs", type=int, default=50)
    ap.add_argument("--b-features", choices=["rgb", "xyz"], default="rgb",
                    help="Use xyz for the shared ScanNet++ semantic data")
    ap.add_argument("--b-lr", type=float, default=0.001)
    ap.add_argument("--b-weight-decay", type=float, default=0.0001)
    ap.add_argument("--b-patience", type=int, default=10,
                    help="Epochs without improvement; 0 disables early stopping")
    ap.add_argument("--b-channels", nargs="+", type=int, default=[32, 64, 128, 256])
    ap.add_argument("--b-task-classes", nargs="+",
                    help="Exact class names for B3, chosen to match your Part E task")
    ap.add_argument("--b-resume", action="store_true",
                    help="Continue the last completed epoch from each last.pth")
    ap.add_argument("--b-evaluate-only", action="store_true",
                    help="Re-evaluate existing best.pth weights without training")
    ap.add_argument("--recon-threshold", type=float, default=0.05,
                    help="One threshold for C3 and D3, so the splat and "
                         "feed-forward numbers are comparable.")
    ap.add_argument("--robot-radius", type=float, default=0.15,
                    help="Part E: radius (m) of the spherical "
                         "robot, the same for every representation and the GT.")
    args = ap.parse_args()

    if not args.data.exists():
        print(f"error: {args.data} does not exist. Prepare the course data first. Note the "
              f"ScanNet++ subset needs you to have accepted the dataset terms, "
              f"which takes time to approve.", file=sys.stderr)
        return 2

    args.out.mkdir(parents=True, exist_ok=True)
    manifest = RunManifest(run_name="lab1", seed=args.seed)
    manifest.add_param("recon_threshold_m", args.recon_threshold)
    manifest.add_param("robot_radius_m", args.robot_radius)

    table = report.lab1_summary_table()
    parts = PARTS if args.part == "all" else [args.part]
    completed, failed = [], []

    for part in parts:
        print(f"\n=== Part {part.upper()} " + "=" * 50)
        try:
            _run_part(part, args, manifest, table)
            completed.append(part)
        except NotImplementedError as e:
            print(f"  not implemented: {e}")
            failed.append(part)
        except Exception:
            print("  FAILED with an exception:")
            traceback.print_exc()
            failed.append(part)

    manifest.add_result("completed_parts", completed)
    manifest.add_result("failed_parts", failed)
    manifest.save(args.out)
    table.conditions = manifest.conditions_line()
    report.write_report_bundle(args.out, [table])
    print("\n" + "=" * 62)
    print(f"completed: {completed or 'none'}   outstanding: {failed or 'none'}")
    print("\nReminder: three pipelines and a PSNR table is the failure mode this "
          "lab is built around. Part E is 40 points and it asks a different "
          "question than Parts B-D do.")
    return 1 if failed else 0


def _run_part(part, args, manifest, table) -> None:
    if part == "b":
        import segmentation  # noqa: F401
        from lab1.part_b import run_part_b

        records = run_part_b(
            args.data, args.out / "part_b", voxel_sizes=args.b_voxel_sizes,
            epochs=args.b_epochs, lr=args.b_lr, weight_decay=args.b_weight_decay,
            patience=None if args.b_patience == 0 else args.b_patience,
            channels=args.b_channels, seed=args.seed,
            task_classes=args.b_task_classes, resume=args.b_resume,
            evaluate_only=args.b_evaluate_only,
            feature_mode=args.b_features,
        )
        manifest.add_param("part_b_dataset", str(args.data.resolve()))
        manifest.add_param("part_b_voxel_sizes_m", args.b_voxel_sizes)
        manifest.add_param("part_b_trials", 1)
        manifest.add_result("part_b", records)
        # Use 5 cm for this summary row when present; detailed results retain all sizes.
        chosen = next((row for row in records if abs(row["voxel_size_m"] - 0.05) < 1e-8), records[0])
        table.rows[0][1] = f"mIoU={chosen['miou']:.4f}, voxel={chosen['voxel_size_m']:g} m"
        table.rows[0][3] = chosen["training_time_s"]
        table.notes += (" Part B fidelity is validation point-level mIoU. Part B memory "
                        "and query latency remain blank until the Part E occupancy "
                        "representation is built; inference measurements are saved separately.")
        return
    if part == "c":
        import splatting  # noqa: F401
        raise NotImplementedError(
            "Part C: fit_splats on held-out views, extract_surface, "
            "evaluate_geometry at --recon-threshold.")
    if part == "d":
        import feedforward  # noqa: F401
        raise NotImplementedError(
            "Part D: run_vggt with NO poses and NO intrinsics, "
            "align_to_ground_truth, evaluate_geometry with its sim3.")
    if part == "e":
        import downstream  # noqa: F401
        raise NotImplementedError(
            "Part E: implement SegmentationOccupancy, SplatOccupancy, "
            "FeedForwardOccupancy and the TSDFOccupancy baseline, "
            "label_ground_truth_occupancy on sample_task_configurations, then evaluate_representations -- "
            "which holds the query procedure fixed, as E2 requires. Finish with "
            "find_ranking_flip and explain the mechanism.")


if __name__ == "__main__":
    raise SystemExit(main())
