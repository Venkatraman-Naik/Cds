#!/usr/bin/env python3
"""
Dataset Inspector & Diagnostic Utility for UCF-Crime
===================================================
Run this on your college machine to verify your 91 GB UCF-Crime dataset
structure, video codecs, and class distributions before starting training.
"""

import os
import sys
import glob
import argparse
from collections import defaultdict
import cv2


def inspect_dataset(dataset_dir):
    print("=" * 65)
    print("  UCF-CRIME DATASET DIAGNOSTIC TOOL")
    print("=" * 65)
    print(f"  Target Path: {os.path.abspath(dataset_dir)}")

    if not os.path.exists(dataset_dir):
        print(f"[ERROR] Directory '{dataset_dir}' does not exist!")
        sys.exit(1)

    # Check top-level subdirectories
    subdirs = [d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))]
    print(f"  Found {len(subdirs)} subdirectories: {', '.join(sorted(subdirs)[:8])}...")

    # Check videos
    video_exts = ('*.mp4', '*.avi', '*.mov', '*.mkv')
    video_files = []
    for ext in video_exts:
        video_files.extend(glob.glob(os.path.join(dataset_dir, "**", ext), recursive=True))

    # Check frames
    img_files = (glob.glob(os.path.join(dataset_dir, "**", "*.jpg"), recursive=True) +
                 glob.glob(os.path.join(dataset_dir, "**", "*.png"), recursive=True))

    category_counts = defaultdict(int)

    if len(video_files) > 0:
        print(f"\n[INFO] Found {len(video_files)} video files.")
        sub_category_counts = defaultdict(int)
        for v in video_files:
            rel = os.path.relpath(v, dataset_dir)
            parts = rel.split(os.sep)
            cat = parts[0]
            category_counts[cat] += 1
            if len(parts) > 2 and parts[-2] != cat:
                sub_category_counts[parts[-2]] += 1
            elif "normal" not in cat.lower():
                sub_category_counts[cat] += 1

        print("\n  Top-Level Dataset Folders:")
        total_normal, total_crime = 0, 0
        for cat, cnt in sorted(category_counts.items()):
            is_norm = "normal" in cat.lower()
            label_type = "NORMAL (0)" if is_norm else "CRIME (1)"
            if is_norm:
                total_normal += cnt
            else:
                total_crime += cnt
            print(f"   - {cat:<36}: {cnt:>5} videos  [{label_type}]")

        print(f"\n  Summary Totals:")
        print(f"   * Crime Videos  (Label 1.0): {total_crime:>5} videos")
        print(f"   * Normal Videos (Label 0.0): {total_normal:>5} videos")
        print(f"   * Total Dataset Size        : {total_crime + total_normal:>5} videos")

        if sub_category_counts:
            print("\n  Crime / Anomaly Specific Sub-Categories Detected:")
            for subcat, cnt in sorted(sub_category_counts.items()):
                print(f"   - {subcat:<25}: {cnt:>4} videos")

        # Test video decode on first 3 videos
        print("\n[INFO] Testing video codec decoding...")
        test_samples = video_files[:3]
        for s in test_samples:
            cap = cv2.VideoCapture(s)
            opened = cap.isOpened()
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            print(f"   - {os.path.basename(s)}: Readable={opened}, {w}x{h} @ {fps:.1f} FPS, {frames} frames")

    elif len(img_files) > 0:
        print(f"\n[INFO] Found {len(img_files)} extracted frame images.")
        for img in img_files[:2000]:
            rel = os.path.relpath(img, dataset_dir)
            cat = rel.split(os.sep)[0]
            category_counts[cat] += 1

        print("\n  Sample Class Breakdown (Frames):")
        for cat, cnt in sorted(category_counts.items()):
            label_type = "NORMAL (0)" if "normal" in cat.lower() else "CRIME (1)"
            print(f"   - {cat:<25}: {cnt:>6} images sampled [{label_type}]")
    else:
        print("\n[WARNING] No video files (.mp4/.avi) or images (.jpg/.png) found in the specified path.")
        print("Please check your folder path and uncompressed dataset location.")

    print("\n" + "=" * 65)
    print("  READY TO TRAIN RECOMMENDATION:")
    print("=" * 65)
    print("Run this command on your college RTX A5000 workstation:")
    print(f"  python train.py --dataset-dir \"{dataset_dir}\" --batch-size 32 --num-workers 8 --epochs 10")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="UCF-Crime Dataset Inspector")
    parser.add_argument("--dataset-dir", type=str, default="./dataset", help="Path to dataset root folder.")
    args = parser.parse_args()
    inspect_dataset(args.dataset_dir)
