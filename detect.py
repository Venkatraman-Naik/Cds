#!/usr/bin/env python3
"""
AI Crime Detection System - End-to-End Inference Engine
======================================================
Automated end-to-end spatio-temporal crime and anomaly detection pipeline.
Processes input video streams from the `inputs/` folder (or CLI argument)
and saves annotated detections and incident alert logs into `outputs/`.
"""

import os
import sys
import time
import json
import math
import glob
import argparse
import collections
from datetime import datetime

import cv2
import numpy as np
import torch
import torchvision.models.video as video_models
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

from model import CrimeDetector3D


def generate_synthetic_demo(output_path="inputs/demo_sample.mp4", num_frames=180, width=640, height=480):
    """Generates a synthetic test video in the inputs folder if no user video exists."""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    print(f"[INFO] Generating synthetic test video at '{output_path}'...")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, 30.0, (width, height))

    for i in range(num_frames):
        frame = np.ones((height, width, 3), dtype=np.uint8) * 25
        # Grid lines
        for y in range(0, height, 40):
            cv2.line(frame, (0, y), (width, y), (45, 45, 45), 1)
        for x in range(0, width, 40):
            cv2.line(frame, (x, 0), (x, height), (45, 45, 45), 1)

        # Subject 1 (Simulated Person 1)
        w1, h1 = 80, 180
        px1 = int(50 + (i * 3) % (width - w1 - 100))
        py1 = int(150 + int(15 * math.sin(i * 0.15)))
        cv2.rectangle(frame, (px1, py1), (px1 + w1, py1 + h1), (60, 160, 60), -1)
        cv2.circle(frame, (px1 + w1 // 2, py1 + 30), 20, (180, 220, 180), -1)
        cv2.putText(frame, "PERSON_1", (px1, py1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 255, 100), 1)

        # Subject 2 (Simulated Person 2)
        w2, h2 = 70, 160
        px2 = int(width - 150 - (i * 2.5) % (width - w2 - 100))
        py2 = int(180 + int(10 * math.cos(i * 0.2)))
        cv2.rectangle(frame, (px2, py2), (px2 + w2, py2 + h2), (160, 60, 60), -1)
        cv2.circle(frame, (px2 + w2 // 2, py2 + 25), 18, (220, 180, 180), -1)
        cv2.putText(frame, "PERSON_2", (px2, py2 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 100, 255), 1)

        # Header watermark
        cv2.putText(frame, "CRIME DETECTION SYSTEM - DEMO INPUT", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 2)
        cv2.putText(frame, f"Frame {i+1}/{num_frames}", (20, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 160, 160), 1)

        out.write(frame)

    out.release()
    print(f"[INFO] Synthetic test video created successfully.")


def resolve_input_video(user_input):
    """
    Finds video source:
    1. If user passed a specific path or webcam index, use it.
    2. Else scan 'inputs/' for video files.
    3. If empty, generate a demo video into 'inputs/demo_sample.mp4'.
    """
    if user_input:
        return user_input

    # Scan inputs/ directory
    supported_exts = ("*.mp4", "*.avi", "*.mov", "*.MOV", "*.mkv", "*.webm")
    found_videos = []
    for ext in supported_exts:
        found_videos.extend(glob.glob(os.path.join("inputs", ext)))

    # Ignore demo video if there are other user videos
    real_videos = [v for v in found_videos if "demo_sample.mp4" not in v]
    if real_videos:
        # Pick the most recently modified video
        chosen = max(real_videos, key=os.path.getmtime)
        print(f"[INFO] Found video in 'inputs/': '{chosen}'")
        return chosen

    if found_videos:
        chosen = found_videos[0]
        print(f"[INFO] Found video in 'inputs/': '{chosen}'")
        return chosen

    # No video found -> generate demo video in inputs/
    demo_path = os.path.join("inputs", "demo_sample.mp4")
    generate_synthetic_demo(demo_path)
    return demo_path


def resolve_weights_path(user_weights):
    """Resolves model weights path, checking default directories."""
    if user_weights and os.path.exists(user_weights):
        return user_weights
    
    candidates = [
        os.path.join("weights", "best_crime_detector.pth"),
        "best_crime_detector.pth"
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return None


def run_detection(args):
    # Set up hardware device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 
                              ('mps' if torch.backends.mps.is_available() else 'cpu'))
    print(f"[INFO] Hardware Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU/Apple Silicon'})")

    # Resolve input source
    input_source = resolve_input_video(args.input)
    is_webcam = input_source.isdigit()
    source_id = int(input_source) if is_webcam else input_source

    cap = cv2.VideoCapture(source_id)
    if not cap.isOpened():
        print(f"[ERROR] Could not open video source '{input_source}'.")
        sys.exit(1)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if not is_webcam else 0

    print(f"[INFO] Video stream initialized: {width}x{height} @ {fps:.1f} FPS "
          f"({f'{total_frames} frames' if total_frames > 0 else 'Live Stream'})")

    # Prepare outputs
    os.makedirs(args.output_dir, exist_ok=True)
    if not args.output:
        base_name = "webcam" if is_webcam else os.path.splitext(os.path.basename(input_source))[0]
        output_video_path = os.path.join(args.output_dir, f"detected_{base_name}.mp4")
        output_json_path = os.path.join(args.output_dir, f"{base_name}_alerts.json")
    else:
        output_video_path = args.output
        base_name = os.path.splitext(os.path.basename(output_video_path))[0]
        output_json_path = os.path.join(args.output_dir, f"{base_name}_alerts.json")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    print(f"[INFO] Output video destination: '{output_video_path}'")

    # Load YOLOv8 Detector
    yolo_model_path = args.yolo_model
    if not os.path.exists(yolo_model_path) and os.path.exists(os.path.join("weights", args.yolo_model)):
        yolo_model_path = os.path.join("weights", args.yolo_model)
    print(f"[INFO] Loading YOLOv8 Person Detector ({yolo_model_path})...")
    yolo_model = YOLO(yolo_model_path)

    # Load DeepSORT Tracker
    print(f"[INFO] Initializing DeepSORT Tracker...")
    tracker = DeepSort(max_age=30, n_init=3, nms_max_overlap=1.0, max_cosine_distance=0.3)

    # Load 3D-CNN Crime Detector
    weights_path = resolve_weights_path(args.weights)
    if weights_path:
        print(f"[INFO] Using trained 3D-CNN weights: '{weights_path}'")
        classifier = CrimeDetector3D(use_pretrained=False, weights_path=weights_path).to(device)
    else:
        print(f"[INFO] No custom weights found at 'weights/best_crime_detector.pth'.")
        print(f"[INFO] Running with pre-trained 3D ResNet-18 video backbone.")
        print(f"[TIP] Train on your college RTX A5000 GPU via `python train.py` to get fine-tuned crime weights.")
        classifier = CrimeDetector3D(use_pretrained=True).to(device)
    classifier.eval()

    # Preprocessing constants (ImageNet)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(device)

    # Tracking state & threat scoring buffers
    track_buffers = {}
    track_frame_counts = {}
    track_threat_scores = {}      # EMA smoothed score per track
    track_consec_high = {}        # consecutive high-score evaluations per track
    track_last_alert_frame = {}   # last frame an incident was logged per track
    incident_alerts = []
    # Consecutive evaluations required to flag a real threat (avoids single-frame spikes)
    CONSEC_REQUIRED = 3
    # Frames between allowed alert logs per track (prevents 386 alerts in 316 frames)
    ALERT_COOLDOWN_FRAMES = 90

    frame_idx = 0
    start_time = time.time()
    active_threat_count = 0

    print("\n" + "=" * 60)
    print(f"  AI Crime Detection Engine Running...")
    print(f"  Threat Alert Threshold: {args.threshold:.2f}")
    print(f"  Temporal Sampling Stride: {args.frame_stride}")
    print(f"  Press 'q' in GUI window to stop.")
    print("=" * 60 + "\n")

    try:
        while cap.isOpened():
            if args.max_frames > 0 and frame_idx >= args.max_frames:
                print(f"[INFO] Reached max-frames limit ({args.max_frames}). Stopping.")
                break

            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1

            # 1. Human Detection with YOLOv8
            results = yolo_model(frame, device=device, verbose=False)
            detections = []
            for r in results:
                for box in r.boxes:
                    cls = int(box.cls[0].item())
                    conf = float(box.conf[0].item())
                    if cls == 0 and conf >= args.conf_thresh:
                        xmin, ymin, xmax, ymax = box.xyxy[0].tolist()
                        detections.append(([xmin, ymin, xmax - xmin, ymax - ymin], conf, 'person'))

            # 2. Multi-Object Tracking with DeepSORT
            tracks = tracker.update_tracks(detections, frame=frame)
            active_threat_count = 0
            current_frame_has_crime = False

            for track in tracks:
                if not track.is_confirmed():
                    continue

                track_id = track.track_id
                ltrb = track.to_ltrb()
                x1, y1, x2, y2 = map(int, ltrb)

                # Clamp bounding box
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(width - 1, x2), min(height - 1, y2)

                if x2 <= x1 or y2 <= y1:
                    continue

                if track_id not in track_buffers:
                    track_buffers[track_id] = collections.deque(maxlen=16)
                    track_frame_counts[track_id] = 0
                    track_threat_scores[track_id] = 0.0
                    track_consec_high[track_id] = 0
                    track_last_alert_frame[track_id] = -999

                track_frame_counts[track_id] += 1

                # Subsample frames using temporal stride with context padding
                if track_frame_counts[track_id] % args.frame_stride == 0:
                    pad_x = int((x2 - x1) * 0.2)
                    pad_y = int((y2 - y1) * 0.2)
                    cx1, cy1 = max(0, x1 - pad_x), max(0, y1 - pad_y)
                    cx2, cy2 = min(width - 1, x2 + pad_x), min(height - 1, y2 + pad_y)
                    crop = frame[cy1:cy2, cx1:cx2]
                    if crop.size > 0:
                        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                        crop_resized = cv2.resize(crop_rgb, (112, 112))
                        tensor = torch.from_numpy(crop_resized).float().to(device) / 255.0
                        tensor = tensor.permute(2, 0, 1) # (3, H, W)
                        tensor = (tensor - mean) / std
                        track_buffers[track_id].append(tensor)

                # 3. 3D-CNN Spatio-Temporal Threat Evaluation
                if len(track_buffers[track_id]) == 16 and (frame_idx % args.eval_interval == 0):
                    clip_tensor = torch.stack(list(track_buffers[track_id]), dim=1) # (3, 16, 112, 112)
                    clip_tensor = clip_tensor.unsqueeze(0) # (1, 3, 16, 112, 112)
                    with torch.no_grad():
                        threat_prob = classifier(clip_tensor).item()

                    prev_score = track_threat_scores.get(track_id, 0.0)
                    # Asymmetric EMA: conservative rising (0.25 weight), faster decay (0.40 weight)
                    if prev_score > 0:
                        if threat_prob > prev_score:
                            track_threat_scores[track_id] = 0.75 * prev_score + 0.25 * threat_prob
                        else:
                            track_threat_scores[track_id] = 0.60 * prev_score + 0.40 * threat_prob
                    else:
                        track_threat_scores[track_id] = threat_prob

                    # Track consecutive high evaluations
                    if track_threat_scores[track_id] >= args.threshold:
                        track_consec_high[track_id] = track_consec_high.get(track_id, 0) + 1
                    else:
                        track_consec_high[track_id] = 0  # reset streak on drop

                score = track_threat_scores.get(track_id, 0.0)
                consec = track_consec_high.get(track_id, 0)

                # Require 3 consecutive high evaluations to confirm a real threat
                is_threat = (score >= args.threshold) and (consec >= CONSEC_REQUIRED)

                if is_threat:
                    active_threat_count += 1
                    current_frame_has_crime = True
                    # Only log once per ALERT_COOLDOWN_FRAMES per track (no spam)
                    last_alert = track_last_alert_frame.get(track_id, -999)
                    if (frame_idx - last_alert) >= ALERT_COOLDOWN_FRAMES:
                        timestamp_str = f"{int(frame_idx / fps // 60):02d}:{int(frame_idx / fps % 60):02d}"
                        incident_alerts.append({
                            "frame": frame_idx,
                            "timestamp": timestamp_str,
                            "track_id": str(track_id),
                            "threat_score": round(score, 4)
                        })
                        track_last_alert_frame[track_id] = frame_idx

                # Determine visuals based on threat score
                if is_threat:
                    box_color = (0, 0, 255) # Red: CRIME ALERT
                    status_text = f"ID:{track_id} CRIME: {score:.1%}"
                elif score >= (args.threshold * 0.7):
                    box_color = (0, 165, 255) # Orange: ELEVATED
                    status_text = f"ID:{track_id} CAUTION: {score:.1%}"
                else:
                    box_color = (0, 255, 0) # Green: NORMAL
                    status_text = f"ID:{track_id} SAFE: {score:.1%}"

                # Draw bounding box
                thickness = 3 if is_threat else 2
                cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, thickness)

                # Label background & text
                t_size, _ = cv2.getTextSize(status_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(frame, (x1, y1 - 22), (x1 + t_size[0] + 8, y1), box_color, -1)
                cv2.putText(frame, status_text, (x1 + 4, y1 - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

                # Mini threat probability gauge bar under bounding box
                bar_w = min(x2 - x1, 100)
                fill_w = int(bar_w * min(score, 1.0))
                cv2.rectangle(frame, (x1, y2 + 4), (x1 + bar_w, y2 + 10), (50, 50, 50), -1)
                cv2.rectangle(frame, (x1, y2 + 4), (x1 + fill_w, y2 + 10), box_color, -1)

            # 4. Global HUD Overlay
            hud_bg = frame.copy()
            cv2.rectangle(hud_bg, (0, 0), (width, 50), (15, 15, 15), -1)
            frame = cv2.addWeighted(hud_bg, 0.75, frame, 0.25, 0)

            # System Status Banner
            elapsed = time.time() - start_time
            current_fps = frame_idx / elapsed if elapsed > 0 else 0

            if current_frame_has_crime:
                banner_color = (0, 0, 255)
                banner_msg = "ALERT: CRIME / SUSPICIOUS BEHAVIOR DETECTED"
            else:
                banner_color = (0, 220, 0)
                banner_msg = "SYSTEM ACTIVE - NO THREATS DETECTED"

            cv2.circle(frame, (20, 25), 8, banner_color, -1)
            cv2.putText(frame, banner_msg, (38, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

            # Telemetry stats
            stats_msg = f"Device: {device.type.upper()} | FPS: {current_fps:.1f} | Active Tracks: {len(tracks)} | Threats: {active_threat_count}"
            s_size, _ = cv2.getTextSize(stats_msg, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            cv2.putText(frame, stats_msg, (width - s_size[0] - 20, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

            # Write frame to output video
            out_writer.write(frame)

            # Display GUI window if enabled
            if not args.no_display:
                cv2.imshow("AI Crime Detection System - Realtime Monitor", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("[INFO] User terminated monitor.")
                    break

            # Progress log
            if frame_idx % 30 == 0 or frame_idx == total_frames:
                pct = f"({(frame_idx/total_frames*100):.1f}%)" if total_frames > 0 else ""
                print(f"  Frame {frame_idx}/{total_frames or '?'} {pct} | FPS: {current_fps:.1f} | Active Threats: {active_threat_count}", end="\r")

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user.")
    finally:
        cap.release()
        out_writer.release()
        if not args.no_display:
            cv2.destroyAllWindows()

    total_time = time.time() - start_time
    avg_fps = frame_idx / total_time if total_time > 0 else 0

    # Save incident log
    with open(output_json_path, 'w') as f:
        json.dump({
            "video_source": input_source,
            "processed_timestamp": datetime.now().isoformat(),
            "total_frames_processed": frame_idx,
            "duration_seconds": round(total_time, 2),
            "average_fps": round(avg_fps, 2),
            "alert_threshold": args.threshold,
            "incident_count": len(incident_alerts),
            "alerts": incident_alerts
        }, f, indent=2)

    print("\n\n" + "=" * 60)
    print(f"  CRIME DETECTION RUN SUMMARY")
    print("=" * 60)
    print(f"  Input Source:        {input_source}")
    print(f"  Annotated Video:     {output_video_path}")
    print(f"  Alerts Report:       {output_json_path}")
    print(f"  Frames Processed:    {frame_idx} in {total_time:.1f}s (Avg {avg_fps:.1f} FPS)")
    print(f"  Incidents Flagged:   {len(incident_alerts)}")
    print("=" * 60 + "\n")


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="AI Crime Detection System - End-to-End Inference Engine",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--input", "--source", dest="input", type=str, default="",
                        help="Path to input video (.mp4, .avi, etc.), camera index (0), or RTSP url. If omitted, checks 'inputs/' directory.")
    parser.add_argument("--output", type=str, default="",
                        help="Specific destination path for annotated output video. Default saves to outputs/detected_<name>.mp4.")
    parser.add_argument("--output-dir", type=str, default="outputs",
                        help="Directory to store annotated videos and incident reports.")
    parser.add_argument("--weights", type=str, default="",
                        help="Path to fine-tuned 3D-CNN weights file. Checks weights/best_crime_detector.pth by default.")
    parser.add_argument("--yolo-model", type=str, default="yolov8n.pt",
                        help="YOLO model path or weight file.")
    parser.add_argument("--threshold", type=float, default=0.65,
                        help="Anomaly/crime probability threshold (0.65 default reduces false positives).")
    parser.add_argument("--conf-thresh", type=float, default=0.5,
                        help="Confidence threshold for human detection.")
    parser.add_argument("--frame-stride", type=int, default=2,
                        help="Temporal sampling stride between frames in the 16-frame 3D-CNN buffer.")
    parser.add_argument("--eval-interval", type=int, default=16,
                        help="Frequency of full 3D-CNN temporal evaluations in frames.")
    parser.add_argument("--max-frames", type=int, default=0,
                        help="Maximum frames to process (0 = full video).")
    parser.add_argument("--no-display", action="store_true",
                        help="Run headlessly without popping up a GUI window.")
    parser.add_argument("--device", type=str, default="",
                        help="Hardware device override ('cuda', 'mps', or 'cpu').")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    run_detection(args)
