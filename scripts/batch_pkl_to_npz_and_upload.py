"""Batch convert all .pkl in datasets/motions_pkl/x2_ultra/ to .npz and upload to W&B.

For each file in PKL_DIR:
  1. Run pkl_to_npz_mujoco.py to produce datasets/motions_npz/<stem>.npz
  2. Upload the npz to W&B registry under collection name <stem>

Usage:
    python scripts/batch_pkl_to_npz_and_upload.py [--skip_existing] [--upload_only] [--convert_only]

Options:
    --skip_existing   Skip conversion if .npz already exists (default: True)
    --upload_only     Skip conversion, only upload existing .npz files
    --convert_only    Skip upload, only convert pkl -> npz
    --pkl_dir         Override input pkl directory
    --npz_dir         Override output npz directory
    --mjcf            Override MJCF model path
    --output_fps      Output FPS (default: 50)
    --project         W&B project name (default: motions_npz)
    --registry_name   W&B registry artifact type (default: motions)
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch pkl→npz conversion + W&B upload")
    parser.add_argument(
        "--pkl_dir",
        type=str,
        default="datasets/motions_pkl/x2_ultra",
        help="Directory containing input .pkl files",
    )
    parser.add_argument(
        "--npz_dir",
        type=str,
        default="datasets/motions_npz",
        help="Directory for output .npz files",
    )
    parser.add_argument(
        "--mjcf",
        type=str,
        default="source/whole_body_tracking/whole_body_tracking/assets/x2_ultra/x2_ultra.xml",
        help="Path to MuJoCo MJCF model",
    )
    parser.add_argument("--output_fps", type=float, default=50.0, help="Output FPS (default: 50)")
    parser.add_argument("--project", type=str, default="motions_npz", help="W&B project name")
    parser.add_argument("--registry_name", type=str, default="motions", help="W&B registry artifact type")
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        default=True,
        help="Skip conversion if .npz already exists (default: True)",
    )
    parser.add_argument(
        "--no_skip_existing",
        dest="skip_existing",
        action="store_false",
        help="Force re-conversion even if .npz exists",
    )
    parser.add_argument("--upload_only", action="store_true", help="Skip conversion, only upload existing .npz")
    parser.add_argument("--convert_only", action="store_true", help="Skip upload, only convert pkl -> npz")
    return parser.parse_args()


def run_cmd(cmd: list[str], label: str) -> bool:
    """Run a subprocess command. Returns True on success."""
    print(f"  $ {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=False)
    if result.returncode != 0:
        print(f"  [ERROR] {label} failed (exit code {result.returncode})")
        return False
    return True


def main():
    args = parse_args()

    pkl_dir = Path(args.pkl_dir)
    npz_dir = Path(args.npz_dir)
    npz_dir.mkdir(parents=True, exist_ok=True)

    pkl_files = sorted(pkl_dir.glob("*.pkl"))
    if not pkl_files:
        print(f"[WARN] No .pkl files found in {pkl_dir}")
        sys.exit(0)

    print(f"Found {len(pkl_files)} .pkl files in {pkl_dir}")
    print(f"Output dir: {npz_dir}\n")

    python = sys.executable
    convert_script = "scripts/pkl_to_npz_mujoco.py"
    upload_script = "scripts/upload_npz.py"

    success_convert: list[str] = []
    success_upload: list[str] = []
    failed_convert: list[str] = []
    failed_upload: list[str] = []
    skipped: list[str] = []

    for i, pkl_path in enumerate(pkl_files, 1):
        stem = pkl_path.stem  # e.g. "walk1_subject1"
        npz_path = npz_dir / f"{stem}.npz"

        print(f"[{i:3d}/{len(pkl_files)}] {stem}")

        # ---- Step 1: Convert pkl -> npz ----
        if args.upload_only:
            if not npz_path.exists():
                print(f"  [SKIP] --upload_only but {npz_path} does not exist, skipping")
                skipped.append(stem)
                continue
        else:
            if args.skip_existing and npz_path.exists():
                print(f"  [SKIP] {npz_path} already exists, skipping conversion")
                skipped.append(stem)
            else:
                ok = run_cmd(
                    [
                        python,
                        convert_script,
                        "--input_pkl", str(pkl_path),
                        "--mjcf", args.mjcf,
                        "--output_npz", str(npz_path),
                        "--output_fps", str(args.output_fps),
                    ],
                    label=f"convert {stem}",
                )
                if ok:
                    success_convert.append(stem)
                else:
                    failed_convert.append(stem)
                    continue  # don't attempt upload if conversion failed

        # ---- Step 2: Upload npz -> W&B ----
        if args.convert_only:
            continue

        ok = run_cmd(
            [
                python,
                upload_script,
                "--npz_path", str(npz_path),
                "--collection_name", stem,
                "--project", args.project,
                "--registry_name", args.registry_name,
            ],
            label=f"upload {stem}",
        )
        if ok:
            success_upload.append(stem)
        else:
            failed_upload.append(stem)

        print()

    # ---- Summary ----
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    if not args.upload_only:
        print(f"Converted:      {len(success_convert)}")
        print(f"Skipped (npz exists): {len(skipped)}")
        print(f"Convert failed: {len(failed_convert)}")
        if failed_convert:
            for name in failed_convert:
                print(f"  - {name}")
    if not args.convert_only:
        print(f"Uploaded:       {len(success_upload)}")
        print(f"Upload failed:  {len(failed_upload)}")
        if failed_upload:
            for name in failed_upload:
                print(f"  - {name}")


if __name__ == "__main__":
    main()
