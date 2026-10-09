#!/usr/bin/env python3
"""
AI Crime Detection System — Web Dashboard
==========================================
A Flask + SocketIO web dashboard that provides:
  - Live CCTV / Webcam / RTSP stream monitoring
  - Uploaded video file analysis
  - Real-time annotated video frames streamed to the browser
  - Incident alert logs with timestamps and threat scores
"""

import os
import sys
import time
import json
import math
import uuid
import threading
import collections
from datetime import datetime

import cv2
import numpy as np
import torch
import base64

from flask import Flask, render_template, request, jsonify, send_from_directory, Response
from flask_socketio import SocketIO, emit

from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
from model import CrimeDetector3D

# ─── App Setup ────────────────────────────────────────────────────────────────
app = Flask(__name__)
app.config['SECRET_KEY'] = 'crime-detection-secret-2026'
app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(__file__), 'uploads')
app.config['OUTPUT_FOLDER'] = os.path.join(os.path.dirname(__file__), 'outputs')
app.config['MAX_CONTENT_LENGTH'] = 2 * 1024 * 1024 * 1024  # 2GB max upload

socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading',
                    max_http_buffer_size=10 * 1024 * 1024)

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)

# ─── Global State ─────────────────────────────────────────────────────────────
active_sessions = {}   # session_id -> detection thread info
device = None
yolo_model = None
classifier = None
models_loaded = False
model_lock = threading.Lock()


def load_models():
    """Load YOLOv8, DeepSORT, and CrimeDetector3D onto GPU once."""
    global device, yolo_model, classifier, models_loaded
    if models_loaded:
        return

    with model_lock:
        if models_loaded:
            return
        print("[DASHBOARD] Loading AI models...")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'
        print(f"[DASHBOARD] Device: {device} ({gpu_name})")

        # YOLOv8
        yolo_path = "yolov8n.pt"
        if not os.path.exists(yolo_path) and os.path.exists(os.path.join("weights", yolo_path)):
            yolo_path = os.path.join("weights", yolo_path)
        yolo_model = YOLO(yolo_path)
        print("[DASHBOARD] YOLOv8 loaded.")

        # Crime Detector 3D
        weights_path = os.path.join("weights", "best_crime_detector.pth")
        if not os.path.exists(weights_path):
            weights_path = "best_crime_detector.pth"

        if os.path.exists(weights_path):
            classifier = CrimeDetector3D(use_pretrained=False, weights_path=weights_path).to(device)
            print(f"[DASHBOARD] CrimeDetector3D loaded from '{weights_path}'.")
        else:
            classifier = CrimeDetector3D(use_pretrained=True).to(device)
            print("[DASHBOARD] CrimeDetector3D loaded with pretrained backbone (no fine-tuned weights).")
        classifier.eval()

        models_loaded = True
        print("[DASHBOARD] All models ready.\n")


# ─── Detection Pipeline ──────────────────────────────────────────────────────
def run_detection_pipeline(session_id, source, is_live=False, threshold=0.5):
    """
    Main detection loop. Runs in a background thread.
    Streams annotated frames back to the browser via SocketIO.
    """
    global active_sessions
    load_models()

    # Open video source
    if source.isdigit():
        cap = cv2.VideoCapture(int(source))
        is_live = True
    elif source.startswith("rtsp://") or source.startswith("http://") or source.startswith("https://"):
        cap = cv2.VideoCapture(source)
        is_live = True
    else:
        cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        socketio.emit('detection_error', {
            'session_id': session_id,
            'error': f"Could not open video source: {source}"
        })
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if not is_live else 0

    # Init tracker per session
    tracker = DeepSort(max_age=30, n_init=3, nms_max_overlap=1.0, max_cosine_distance=0.3)

    # Preprocessing
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(device)

    # Buffers
    track_buffers = {}
    track_frame_counts = {}
    track_threat_scores = {}      # EMA smoothed score per track
    track_consec_high = {}        # consecutive high-score evaluations per track
    track_last_alert_frame = {}   # last frame an incident was logged per track
    incident_alerts = []

    frame_idx = 0
    start_time = time.time()
    conf_thresh = 0.5
    frame_stride = 2
    eval_interval = 16
    # Number of consecutive high evaluations required before flagging
    CONSEC_REQUIRED = 3
    # Minimum frames between incident log entries for the same track (cooldown)
    ALERT_COOLDOWN_FRAMES = 90

    # Output video writer
    base_name = os.path.splitext(os.path.basename(source))[0] if not is_live else "live_session"
    output_video_path = os.path.join(app.config['OUTPUT_FOLDER'], f"detected_{base_name}_{session_id[:8]}.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    # Update session state
    active_sessions[session_id] = {
        'status': 'running',
        'source': source,
        'is_live': is_live,
        'frame_idx': 0,
        'total_frames': total_frames,
        'incidents': 0,
        'stop_flag': threading.Event()
    }

    socketio.emit('detection_started', {
        'session_id': session_id,
        'source': source,
        'is_live': is_live,
        'width': width,
        'height': height,
        'fps': fps,
        'total_frames': total_frames
    })

    try:
        while cap.isOpened():
            # Check stop flag
            if active_sessions.get(session_id, {}).get('stop_flag', threading.Event()).is_set():
                break

            ret, frame = cap.read()
            if not ret:
                if is_live:
                    time.sleep(0.1)
                    continue
                break

            frame_idx += 1

            # YOLOv8 person detection
            results = yolo_model(frame, device=device, verbose=False)
            detections = []
            for r in results:
                for box in r.boxes:
                    cls = int(box.cls[0].item())
                    conf = float(box.conf[0].item())
                    if cls == 0 and conf >= conf_thresh:
                        xmin, ymin, xmax, ymax = box.xyxy[0].tolist()
                        detections.append(([xmin, ymin, xmax - xmin, ymax - ymin], conf, 'person'))

            # DeepSORT tracking
            tracks = tracker.update_tracks(detections, frame=frame)
            active_threat_count = 0
            current_frame_has_crime = False

            for track in tracks:
                if not track.is_confirmed():
                    continue

                track_id = track.track_id
                ltrb = track.to_ltrb()
                x1, y1, x2, y2 = map(int, ltrb)
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

                # Subsample with spatial context padding
                if track_frame_counts[track_id] % frame_stride == 0:
                    pad_x = int((x2 - x1) * 0.2)
                    pad_y = int((y2 - y1) * 0.2)
                    cx1, cy1 = max(0, x1 - pad_x), max(0, y1 - pad_y)
                    cx2, cy2 = min(width - 1, x2 + pad_x), min(height - 1, y2 + pad_y)
                    crop = frame[cy1:cy2, cx1:cx2]
                    if crop.size > 0:
                        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                        crop_resized = cv2.resize(crop_rgb, (112, 112))
                        tensor = torch.from_numpy(crop_resized).float().to(device) / 255.0
                        tensor = tensor.permute(2, 0, 1)
                        tensor = (tensor - mean) / std
                        track_buffers[track_id].append(tensor)

                # 3D-CNN evaluation with smart EMA + decay
                if len(track_buffers[track_id]) == 16 and (frame_idx % eval_interval == 0):
                    clip_tensor = torch.stack(list(track_buffers[track_id]), dim=1)
                    clip_tensor = clip_tensor.unsqueeze(0)
                    with torch.no_grad():
                        threat_prob = classifier(clip_tensor).item()

                    prev_score = track_threat_scores.get(track_id, 0.0)
                    if prev_score > 0:
                        # Asymmetric EMA: slow to rise (0.25 new weight), fast to decay
                        if threat_prob > prev_score:
                            # Rising: weight new observation at 25% (conservative)
                            new_score = 0.75 * prev_score + 0.25 * threat_prob
                        else:
                            # Falling: decay faster at 40% new weight (clear faster)
                            new_score = 0.60 * prev_score + 0.40 * threat_prob
                        track_threat_scores[track_id] = new_score
                    else:
                        track_threat_scores[track_id] = threat_prob

                    # Count consecutive evaluations above threshold
                    if track_threat_scores[track_id] >= threshold:
                        track_consec_high[track_id] = track_consec_high.get(track_id, 0) + 1
                    else:
                        # Reset streak when score drops below threshold
                        track_consec_high[track_id] = 0

                score = track_threat_scores.get(track_id, 0.0)
                consec = track_consec_high.get(track_id, 0)

                # Only flag as threat if CONSEC_REQUIRED consecutive evaluations are high
                is_threat = (score >= threshold) and (consec >= CONSEC_REQUIRED)

                if is_threat:
                    active_threat_count += 1
                    current_frame_has_crime = True
                    # Only log once per ALERT_COOLDOWN_FRAMES per track to avoid spam
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

                # Visual overlays
                if is_threat:
                    box_color = (0, 0, 255)   # Red
                    status_text = f"ID:{track_id} CRIME: {score:.0%}"
                    thickness = 3
                elif score >= (threshold * 0.7):
                    box_color = (0, 165, 255)  # Orange
                    status_text = f"ID:{track_id} CAUTION: {score:.0%}"
                    thickness = 2
                else:
                    box_color = (0, 255, 0)    # Green
                    status_text = f"ID:{track_id} SAFE"
                    thickness = 1

                # Only draw boxes for threats / caution; normal gets thin green outline
                cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, thickness)

                # Label
                t_size, _ = cv2.getTextSize(status_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(frame, (x1, y1 - 22), (x1 + t_size[0] + 8, y1), box_color, -1)
                cv2.putText(frame, status_text, (x1 + 4, y1 - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

                # Mini threat bar
                bar_w = min(x2 - x1, 100)
                fill_w = int(bar_w * min(score, 1.0))
                cv2.rectangle(frame, (x1, y2 + 4), (x1 + bar_w, y2 + 10), (50, 50, 50), -1)
                cv2.rectangle(frame, (x1, y2 + 4), (x1 + fill_w, y2 + 10), box_color, -1)

            # HUD overlay
            hud_bg = frame.copy()
            cv2.rectangle(hud_bg, (0, 0), (width, 50), (15, 15, 15), -1)
            frame = cv2.addWeighted(hud_bg, 0.75, frame, 0.25, 0)

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

            stats_msg = f"FPS: {current_fps:.1f} | Tracks: {len(tracks)} | Threats: {active_threat_count}"
            s_size, _ = cv2.getTextSize(stats_msg, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            cv2.putText(frame, stats_msg, (width - s_size[0] - 20, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

            # Write output
            out_writer.write(frame)

            # Encode frame and send to browser (resize for bandwidth)
            display_w = min(width, 960)
            display_h = int(height * (display_w / width))
            display_frame = cv2.resize(frame, (display_w, display_h))
            _, buffer = cv2.imencode('.jpg', display_frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
            frame_b64 = base64.b64encode(buffer).decode('utf-8')

            # Compute progress
            progress = (frame_idx / total_frames * 100) if total_frames > 0 else 0

            socketio.emit('detection_frame', {
                'session_id': session_id,
                'frame': frame_b64,
                'frame_idx': frame_idx,
                'total_frames': total_frames,
                'progress': round(progress, 1),
                'fps': round(current_fps, 1),
                'active_threats': active_threat_count,
                'has_crime': current_frame_has_crime,
                'incidents_total': len(incident_alerts)
            })

            # Update session
            active_sessions[session_id]['frame_idx'] = frame_idx
            active_sessions[session_id]['incidents'] = len(incident_alerts)

            # Throttle to ~15fps for streaming
            if not is_live:
                time.sleep(max(0, (1.0 / 15.0) - (time.time() - start_time - elapsed)))

    except Exception as e:
        socketio.emit('detection_error', {
            'session_id': session_id,
            'error': str(e)
        })
    finally:
        cap.release()
        out_writer.release()

    total_time = time.time() - start_time

    # Save incident report
    report_path = os.path.join(app.config['OUTPUT_FOLDER'], f"{base_name}_{session_id[:8]}_alerts.json")
    with open(report_path, 'w') as f:
        json.dump({
            "video_source": source,
            "processed_timestamp": datetime.now().isoformat(),
            "total_frames_processed": frame_idx,
            "duration_seconds": round(total_time, 2),
            "alert_threshold": threshold,
            "incident_count": len(incident_alerts),
            "alerts": incident_alerts
        }, f, indent=2)

    # Notify completion
    socketio.emit('detection_complete', {
        'session_id': session_id,
        'total_frames': frame_idx,
        'duration': round(total_time, 1),
        'incidents': len(incident_alerts),
        'output_video': os.path.basename(output_video_path),
        'report': os.path.basename(report_path),
        'alerts': incident_alerts[-20:]  # Last 20 alerts for display
    })

    # Cleanup session
    if session_id in active_sessions:
        active_sessions[session_id]['status'] = 'completed'


# ─── Flask Routes ─────────────────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_video():
    """Handle video file upload."""
    if 'video' not in request.files:
        return jsonify({'error': 'No video file provided'}), 400

    file = request.files['video']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    # Save uploaded file
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in ['.mp4', '.avi', '.mkv', '.mov', '.wmv', '.flv', '.webm']:
        return jsonify({'error': 'Unsupported video format. Use MP4, AVI, MKV, MOV.'}), 400

    safe_name = f"{uuid.uuid4().hex[:12]}{ext}"
    save_path = os.path.join(app.config['UPLOAD_FOLDER'], safe_name)
    file.save(save_path)

    return jsonify({
        'success': True,
        'filename': safe_name,
        'original_name': file.filename,
        'path': save_path
    })


@app.route('/outputs/<path:filename>')
def serve_output(filename):
    """Serve processed output files."""
    return send_from_directory(app.config['OUTPUT_FOLDER'], filename)


@app.route('/api/status')
def api_status():
    """Return system status."""
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'
    vram = f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB" if torch.cuda.is_available() else 'N/A'
    weights_exists = os.path.exists(os.path.join("weights", "best_crime_detector.pth"))
    return jsonify({
        'gpu': gpu_name,
        'vram': vram,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'models_loaded': models_loaded,
        'weights_trained': weights_exists,
        'active_sessions': len([s for s in active_sessions.values() if s.get('status') == 'running'])
    })


# ─── SocketIO Events ─────────────────────────────────────────────────────────
@socketio.on('start_detection')
def handle_start_detection(data):
    """Start a detection session."""
    mode = data.get('mode', 'video')  # 'video' or 'live'
    source = data.get('source', '')
    session_id = str(uuid.uuid4())

    if mode == 'live':
        # Camera index or RTSP URL
        if not source:
            source = '0'  # Default webcam
    elif mode == 'video':
        # File path from upload
        if not source:
            emit('detection_error', {'session_id': session_id, 'error': 'No video source specified'})
            return
        # Resolve relative to uploads
        if not os.path.isabs(source):
            source = os.path.join(app.config['UPLOAD_FOLDER'], source)
        if not os.path.exists(source):
            emit('detection_error', {'session_id': session_id, 'error': f'Video file not found: {source}'})
            return

    user_threshold = float(data.get('threshold', 0.5))

    # Launch detection in background thread
    thread = threading.Thread(
        target=run_detection_pipeline,
        args=(session_id, source, mode == 'live', user_threshold),
        daemon=True
    )
    thread.start()

    emit('session_created', {'session_id': session_id, 'mode': mode, 'source': source})


@socketio.on('stop_detection')
def handle_stop_detection(data):
    """Stop a running detection session."""
    session_id = data.get('session_id', '')
    if session_id in active_sessions:
        active_sessions[session_id]['stop_flag'].set()
        active_sessions[session_id]['status'] = 'stopped'
        emit('detection_stopped', {'session_id': session_id})


@socketio.on('connect')
def handle_connect():
    print(f"[DASHBOARD] Client connected: {request.sid}")
    # Pre-load models on first connection
    threading.Thread(target=load_models, daemon=True).start()


@socketio.on('disconnect')
def handle_disconnect():
    print(f"[DASHBOARD] Client disconnected: {request.sid}")


# ─── Entry Point ──────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("  AI CRIME DETECTION SYSTEM — WEB DASHBOARD")
    print("=" * 60)
    print("  Open in browser: http://localhost:5000")
    print("=" * 60 + "\n")
    socketio.run(app, host='0.0.0.0', port=5000, debug=False, allow_unsafe_werkzeug=True)
