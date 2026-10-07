#!/usr/bin/env python3
"""
AI Crime Detection System - High-Performance Training Pipeline
==============================================================
Optimized for training on the 91 GB UCF-Crime dataset using high-end
NVIDIA GPUs (e.g. NVIDIA RTX A5000 with 24GB VRAM) and PyTorch CUDA acceleration.

Features:
- Automatic Mixed Precision (torch.cuda.amp) for 2-3x speedup on Ampere/RTX tensor cores.
- Dual dataset support: handles raw video files (.mp4/.avi) OR extracted frame directories (.jpg/.png).
- Multi-worker DataLoader with pinned memory and persistent workers.
- Video-level train/validation split to prevent temporal data leakage.
- Automatic checkpointing to 'weights/best_crime_detector.pth' and resume capability.
- Full validation metrics: Loss, Accuracy, Precision, Recall, and F1 score.
"""

import os
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
import glob
import time
import json
import random
import argparse
from collections import defaultdict

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from model import CrimeDetector3D


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = True


class VideoClipDataset(Dataset):
    """
    Direct Video File Loader:
    Extracts 16-frame sub-sequences on-the-fly from raw video files (.mp4/.avi)
    without requiring hundreds of gigabytes of pre-extracted frame images.
    """
    def __init__(self, clip_entries, seq_len=16, is_train=True, img_size=(112, 112)):
        self.clip_entries = clip_entries # list of (video_path, start_frame, frame_stride, label)
        self.seq_len = seq_len
        self.is_train = is_train
        self.img_size = img_size
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)

    def __len__(self):
        return len(self.clip_entries)

    def __getitem__(self, idx):
        video_path, start_frame, stride, label = self.clip_entries[idx]
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        do_flip = self.is_train and (random.random() > 0.5)
        frames = []

        for step in range(self.seq_len):
            ret, frame = cap.read()
            if not ret or frame is None:
                # If reading beyond EOF, pad with last frame or black
                frame = frames[-1] if frames else np.zeros((self.img_size[1], self.img_size[0], 3), dtype=np.uint8)
            else:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, self.img_size)

            if do_flip:
                frame = cv2.flip(frame, 1)

            # Skip forward if stride > 1
            if stride > 1 and step < self.seq_len - 1:
                current_pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
                cap.set(cv2.CAP_PROP_POS_FRAMES, current_pos + stride - 1)

            # Normalize
            norm = (frame.astype(np.float32) / 255.0 - self.mean) / self.std
            frames.append(norm.transpose(2, 0, 1))

        cap.release()

        # Stack into (T, C, H, W) then transpose to (C, T, H, W)
        seq_tensor = np.stack(frames, axis=0).transpose(1, 0, 2, 3)
        return torch.tensor(seq_tensor, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)


class FrameSequenceDataset(Dataset):
    """
    Extracted Frames Loader:
    Loads 16-frame sequences from pre-extracted image files (.jpg/.png).
    """
    def __init__(self, sequences, seq_len=16, is_train=True, img_size=(112, 112)):
        self.sequences = sequences # list of (frame_paths, label)
        self.seq_len = seq_len
        self.is_train = is_train
        self.img_size = img_size
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        frame_paths, label = self.sequences[idx]
        do_flip = self.is_train and (random.random() > 0.5)

        frames = []
        for path in frame_paths:
            img = cv2.imread(path)
            if img is None:
                img = np.zeros((self.img_size[1], self.img_size[0], 3), dtype=np.uint8)
            else:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, self.img_size)

            if do_flip:
                img = cv2.flip(img, 1)

            norm = (img.astype(np.float32) / 255.0 - self.mean) / self.std
            frames.append(norm.transpose(2, 0, 1))

        seq_tensor = np.stack(frames, axis=0).transpose(1, 0, 2, 3)
        return torch.tensor(seq_tensor, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)


def discover_dataset(dataset_dir, seq_len=16, stride=32, clip_sample_stride=2, max_clips_per_video=None):
    """
    Intelligently inspects the dataset folder and auto-detects format:
    1. Raw video files (.mp4, .avi)
    2. Extracted frame images (.jpg, .png)
    """
    if not os.path.exists(dataset_dir):
        raise FileNotFoundError(f"Dataset directory '{dataset_dir}' does not exist!")

    print(f"\n[DATASET] Scanning root directory: '{dataset_dir}'...")
    subdirs = [d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))]

    # Check whether directory contains video files
    video_exts = ('*.mp4', '*.avi', '*.mov', '*.mkv')
    sample_videos = []
    for ext in video_exts:
        sample_videos.extend(glob.glob(os.path.join(dataset_dir, "**", ext), recursive=True))

    if len(sample_videos) > 0:
        print(f"[DATASET] Detected raw video dataset mode ({len(sample_videos)} video files found).")
        return "video", parse_video_dataset(dataset_dir, sample_videos, seq_len, stride, clip_sample_stride, max_clips_per_video)

    # Otherwise check for frame image files
    print(f"[DATASET] Scanning for extracted frame image files (.jpg / .png)...")
    return "frames", parse_frame_dataset(dataset_dir, subdirs, seq_len, stride, max_clips_per_video)


def parse_video_dataset(dataset_dir, video_files, seq_len, stride, clip_sample_stride, max_clips_per_video):
    """Parses raw video files into training clip metadata."""
    clip_entries = []
    video_entries_by_id = defaultdict(list)
    normal_count, crime_count = 0, 0

    print(f"[DATASET] Indexing {len(video_files)} video files...")
    for idx, video_path in enumerate(video_files):
        rel_path = os.path.relpath(video_path, dataset_dir)
        category = rel_path.split(os.sep)[0].lower()
        is_normal = ("normal" in category)
        label = 0.0 if is_normal else 1.0
        video_id = os.path.splitext(os.path.basename(video_path))[0]

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            continue
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        needed_span = seq_len * clip_sample_stride
        if total_frames < needed_span:
            # Short video -> single clip starting at 0
            entry = (video_path, 0, clip_sample_stride, label, video_id)
            video_entries_by_id[video_id].append(entry)
            if is_normal: normal_count += 1
            else: crime_count += 1
        else:
            clips_added = 0
            for start_frame in range(0, total_frames - needed_span + 1, stride):
                if max_clips_per_video and clips_added >= max_clips_per_video:
                    break
                entry = (video_path, start_frame, clip_sample_stride, label, video_id)
                video_entries_by_id[video_id].append(entry)
                clips_added += 1
                if is_normal: normal_count += 1
                else: crime_count += 1

        if (idx + 1) % 100 == 0 or (idx + 1) == len(video_files):
            print(f"  Indexed {idx + 1}/{len(video_files)} videos...", end="\r")

    print(f"\n[DATASET] Indexing complete: {len(video_entries_by_id)} unique videos.")
    print(f"[DATASET] Total clips extracted: {normal_count + crime_count} (Normal: {normal_count}, Crime: {crime_count})")
    return video_entries_by_id


def parse_frame_dataset(dataset_dir, subdirs, seq_len, stride, max_clips_per_video):
    """Parses extracted frame images (UCF-Crime frame format)."""
    video_frames = defaultdict(list)
    video_labels = {}

    for cat in subdirs:
        cat_path = os.path.join(dataset_dir, cat)
        is_normal = ("normal" in cat.lower())
        label = 0.0 if is_normal else 1.0

        image_files = (glob.glob(os.path.join(cat_path, '*.png')) + 
                       glob.glob(os.path.join(cat_path, '*.jpg')) +
                       glob.glob(os.path.join(cat_path, '*/*.jpg')) +
                       glob.glob(os.path.join(cat_path, '*/*.png')))

        for filepath in image_files:
            filename = os.path.basename(filepath)
            parts = filename.rsplit('_', 1)
            if len(parts) == 2:
                video_id = parts[0]
                idx_str = os.path.splitext(parts[1])[0]
                try:
                    frame_idx = int(idx_str)
                    video_frames[video_id].append((frame_idx, filepath))
                    video_labels[video_id] = label
                except ValueError:
                    continue
            else:
                # Subfolder organization
                parent = os.path.basename(os.path.dirname(filepath))
                video_id = parent
                idx_str = os.path.splitext(filename)[0]
                try:
                    frame_idx = int(idx_str)
                except ValueError:
                    frame_idx = len(video_frames[video_id])
                video_frames[video_id].append((frame_idx, filepath))
                video_labels[video_id] = label

    video_entries_by_id = defaultdict(list)
    normal_count, crime_count = 0, 0

    for video_id, items in video_frames.items():
        items.sort(key=lambda x: x[0])
        paths = [item[1] for item in items]
        label = video_labels.get(video_id, 0.0)

        if len(paths) < seq_len:
            padded = paths + [paths[-1]] * (seq_len - len(paths))
            video_entries_by_id[video_id].append((padded, label, video_id))
            if label == 0.0: normal_count += 1
            else: crime_count += 1
        else:
            clips_added = 0
            for i in range(0, len(paths) - seq_len + 1, stride):
                if max_clips_per_video and clips_added >= max_clips_per_video:
                    break
                seq = paths[i:i + seq_len]
                video_entries_by_id[video_id].append((seq, label, video_id))
                clips_added += 1
                if label == 0.0: normal_count += 1
                else: crime_count += 1

    print(f"\n[DATASET] Extracted frame indexing complete: {len(video_frames)} unique videos.")
    print(f"[DATASET] Total sequences extracted: {normal_count + crime_count} (Normal: {normal_count}, Crime: {crime_count})")
    return video_entries_by_id


def train():
    parser = argparse.ArgumentParser(
        description="High-Performance 3D-CNN Crime Detection Training (RTX A5000 / CUDA Optimized)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--dataset-dir", type=str, default="./dataset",
                        help="Path to the 91 GB UCF-Crime dataset directory.")
    parser.add_argument("--epochs", type=int, default=10,
                        help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Batch size (e.g. 32 or 64 recommended for 24GB RTX A5000).")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate for AdamW.")
    parser.add_argument("--weight-decay", type=float, default=1e-4,
                        help="Weight decay.")
    parser.add_argument("--seq-len", type=int, default=16,
                        help="Number of frames per 3D-CNN temporal sequence.")
    parser.add_argument("--stride", type=int, default=32,
                        help="Stride between sequence start frames.")
    parser.add_argument("--clip-sample-stride", type=int, default=2,
                        help="Stride between sampled frames within a clip.")
    parser.add_argument("--max-clips-per-video", type=int, default=0,
                        help="Cap clips per video to balance dataset size (0 = unlimited).")
    parser.add_argument("--val-split", type=float, default=0.2,
                        help="Fraction of videos reserved for validation.")
    parser.add_argument("--num-workers", type=int, default=8,
                        help="Number of DataLoader worker processes (8 recommended for workstation).")
    parser.add_argument("--save-dir", type=str, default="weights",
                        help="Directory to save model checkpoints.")
    parser.add_argument("--resume", type=str, default="",
                        help="Path to checkpoint to resume training from.")
    parser.add_argument("--no-amp", action="store_true",
                        help="Disable Automatic Mixed Precision (AMP). By default, AMP is enabled on CUDA.")
    args = parser.parse_args()

    set_seed(42)

    # 1. Device and Acceleration Setup
    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else ('mps' if torch.backends.mps.is_available() else 'cpu'))
    use_amp = use_cuda and (not args.no_amp)

    print("\n" + "=" * 65)
    print("  3D-CNN CRIME DETECTION SYSTEM - WORKSTATION TRAINING")
    print("=" * 65)
    print(f"  Target Device:       {device}")
    if use_cuda:
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"  GPU Hardware:        {gpu_name} ({gpu_mem:.1f} GB VRAM)")
        print(f"  Mixed Precision:     {'Enabled (FP16 Tensor Cores)' if use_amp else 'Disabled'}")
    print(f"  Batch Size:          {args.batch_size}")
    print(f"  Epochs:              {args.epochs}")
    print(f"  DataLoader Workers:  {args.num_workers}")
    print("=" * 65)

    os.makedirs(args.save_dir, exist_ok=True)
    best_weights_path = os.path.join(args.save_dir, "best_crime_detector.pth")
    last_checkpoint_path = os.path.join(args.save_dir, "last_checkpoint.pth")

    # 2. Dataset Discovery & Split
    max_clips = args.max_clips_per_video if args.max_clips_per_video > 0 else None
    dataset_mode, video_dict = discover_dataset(
        args.dataset_dir,
        seq_len=args.seq_len,
        stride=args.stride,
        clip_sample_stride=args.clip_sample_stride,
        max_clips_per_video=max_clips
    )

    video_ids = list(video_dict.keys())
    random.shuffle(video_ids)
    num_val = int(len(video_ids) * args.val_split)
    val_vids = set(video_ids[:num_val])
    train_vids = set(video_ids[num_val:])

    train_entries = []
    val_entries = []

    for vid in video_ids:
        entries = video_dict[vid]
        target = val_entries if vid in val_vids else train_entries
        for e in entries:
            # Tuple formats:
            # Video mode: (video_path, start_frame, stride, label, vid) -> need (path, start, stride, label)
            # Frame mode: (paths, label, vid) -> need (paths, label)
            if dataset_mode == "video":
                target.append((e[0], e[1], e[2], e[3]))
            else:
                target.append((e[0], e[1]))

    print(f"\n[SPLIT] Training Set:   {len(train_vids)} videos | {len(train_entries)} clips")
    print(f"[SPLIT] Validation Set: {len(val_vids)} videos | {len(val_entries)} clips")

    if dataset_mode == "video":
        train_ds = VideoClipDataset(train_entries, seq_len=args.seq_len, is_train=True)
        val_ds = VideoClipDataset(val_entries, seq_len=args.seq_len, is_train=False)
    else:
        train_ds = FrameSequenceDataset(train_entries, seq_len=args.seq_len, is_train=True)
        val_ds = FrameSequenceDataset(val_entries, seq_len=args.seq_len, is_train=False)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=use_cuda,
        persistent_workers=(args.num_workers > 0)
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=use_cuda,
        persistent_workers=(args.num_workers > 0)
    )

    # 3. Model, Loss, Optimizer, Scaler
    model = CrimeDetector3D(use_pretrained=True).to(device)
    criterion = nn.BCELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    if hasattr(torch.amp, 'GradScaler'):
        scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
    else:
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    start_epoch = 1
    best_val_loss = float('inf')
    best_val_acc = 0.0

    # Resume from checkpoint if provided
    if args.resume and os.path.exists(args.resume):
        print(f"[RESUME] Loading checkpoint from '{args.resume}'...")
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        start_epoch = ckpt.get('epoch', 0) + 1
        best_val_loss = ckpt.get('val_loss', float('inf'))
        best_val_acc = ckpt.get('val_acc', 0.0)
        print(f"[RESUME] Resuming from Epoch {start_epoch} (Best Val Loss: {best_val_loss:.4f})")

    # 4. Training Loop
    print("\n" + "=" * 65)
    print("  TRAINING STARTED")
    print("=" * 65)

    for epoch in range(start_epoch, args.epochs + 1):
        epoch_start = time.time()

        # Training Phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            optimizer.zero_grad()

            if hasattr(torch.amp, 'autocast'):
                autocast_ctx = torch.amp.autocast('cuda', enabled=use_amp)
            else:
                autocast_ctx = torch.cuda.amp.autocast(enabled=use_amp)

            with autocast_ctx:
                outputs = model(inputs).squeeze(1)

            # Compute BCELoss in FP32 outside autocast to prevent float16 underflow/overflow
            outputs = torch.clamp(outputs.float(), 1e-7, 1.0 - 1e-7)
            loss = criterion(outputs, targets)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item() * inputs.size(0)
            preds = (outputs >= 0.5).float()
            train_correct += (preds == targets).sum().item()
            train_total += targets.size(0)

            if (batch_idx + 1) % 20 == 0 or (batch_idx + 1) == len(train_loader):
                cur_acc = (train_correct / train_total) * 100
                cur_loss = train_loss / train_total
                pct = (batch_idx + 1) / len(train_loader) * 100
                print(f"  [Epoch {epoch:02d}/{args.epochs:02d} | Step {batch_idx+1:04d}/{len(train_loader):04d} ({pct:4.1f}%)] "
                      f"Loss: {cur_loss:.4f} | Acc: {cur_acc:.2f}%", flush=True)

        scheduler.step()
        epoch_train_loss = train_loss / train_total if train_total > 0 else 0.0
        epoch_train_acc = train_correct / train_total if train_total > 0 else 0.0

        # Validation Phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        tp, fp, fn, tn = 0, 0, 0, 0

        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = inputs.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)

                if hasattr(torch.amp, 'autocast'):
                    autocast_ctx = torch.amp.autocast('cuda', enabled=use_amp)
                else:
                    autocast_ctx = torch.cuda.amp.autocast(enabled=use_amp)

                with autocast_ctx:
                    outputs = model(inputs).squeeze(1)

                outputs = torch.clamp(outputs.float(), 1e-7, 1.0 - 1e-7)
                loss = criterion(outputs, targets)

                val_loss += loss.item() * inputs.size(0)
                preds = (outputs >= 0.5).float()
                val_correct += (preds == targets).sum().item()
                val_total += targets.size(0)

                tp += ((preds == 1.0) & (targets == 1.0)).sum().item()
                fp += ((preds == 1.0) & (targets == 0.0)).sum().item()
                fn += ((preds == 0.0) & (targets == 1.0)).sum().item()
                tn += ((preds == 0.0) & (targets == 0.0)).sum().item()

        epoch_val_loss = val_loss / val_total if val_total > 0 else 0.0
        epoch_val_acc = val_correct / val_total if val_total > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        elapsed = time.time() - epoch_start
        print("\n" + "-" * 65)
        print(f"  Epoch {epoch:02d}/{args.epochs:02d} Summary ({elapsed:.1f}s | {elapsed/60:.1f}m):")
        print(f"  Train Loss: {epoch_train_loss:.4f}  |  Train Acc: {epoch_train_acc*100:.2f}%")
        print(f"  Val Loss:   {epoch_val_loss:.4f}  |  Val Acc:   {epoch_val_acc*100:.2f}%")
        print(f"  Precision:  {precision:.4f}  |  Recall:    {recall:.4f}  |  F1: {f1:.4f}")
        print("-" * 65 + "\n")

        # Save Latest Checkpoint
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': epoch_val_loss,
            'val_acc': epoch_val_acc,
        }, last_checkpoint_path)

        # Save Best Model Checkpoint
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            best_val_acc = epoch_val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': best_val_loss,
                'val_acc': best_val_acc,
            }, best_weights_path)
            print(f"  >>> [BEST] NEW BEST MODEL SAVED to '{best_weights_path}' (Val Loss: {best_val_loss:.4f}, Val Acc: {best_val_acc*100:.2f}%)\n")

    print("\n" + "=" * 65)
    print("  TRAINING COMPLETE!")
    print(f"  Best Val Loss: {best_val_loss:.4f} | Best Val Acc: {best_val_acc*100:.2f}%")
    print(f"  Optimal Weights Saved At: '{best_weights_path}'")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    train()
