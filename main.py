"""
main.py
=======
Smart Basketball Training System – main entry point.

Usage
-----
    python main.py [--video PATH] [--save-video] [--detector yolo|classical]

Defaults:
    --video     resrc/testSub1.mp4
    --detector  yolo   (falls back to classical automatically if model unavailable)
    --save-video        flag; if set, writes output/annotated_<name>.mp4
"""

import argparse
import json
import os
import sys

import cv2
import numpy as np

# ── Project modules ───────────────────────────────────────────────────────────
from config import (
    DISPLAY_WIDTH, DISPLAY_HEIGHT,
    TRAIL_MAX_POINTS,
    DETECTOR_TYPE,
)
from ball_detection import BallDetector
from rim_detection   import get_rim_position
from trajectory      import BallTracker
from shot_analysis   import ShotAnalyser


# ──────────────────────────────────────────────────────────────────────────────
# Argument parsing
# ──────────────────────────────────────────────────────────────────────────────

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Smart Basketball Training System",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--video", "-v",
        default=os.path.join(SCRIPT_DIR, "resrc", "testSub1.mp4"),
        help="Path to the input video file.",
    )
    p.add_argument(
        "--save-video", "-s",
        action="store_true",
        help="Write annotated output video to output/annotated_<name>.mp4",
    )
    p.add_argument(
        "--detector", "-d",
        choices=["yolo", "classical"],
        default=DETECTOR_TYPE,
        help="Detection engine to use.",
    )
    return p.parse_args()


# ──────────────────────────────────────────────────────────────────────────────
# Letterbox helper – resize while preserving aspect ratio, no distortion
# ──────────────────────────────────────────────────────────────────────────────

def letterbox(frame: np.ndarray,
              target_w: int = DISPLAY_WIDTH,
              target_h: int = DISPLAY_HEIGHT) -> np.ndarray:
    h0, w0 = frame.shape[:2]
    scale   = min(target_w / w0, target_h / h0)
    nw, nh  = int(w0 * scale), int(h0 * scale)
    resized = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas  = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    y_off   = (target_h - nh) // 2
    x_off   = (target_w - nw) // 2
    canvas[y_off: y_off + nh, x_off: x_off + nw] = resized
    return canvas


# ──────────────────────────────────────────────────────────────────────────────
# Drawing helpers
# ──────────────────────────────────────────────────────────────────────────────

# Phase → BGR colour for the trajectory trail
_PHASE_COLOR = {
    "held":      (120, 120, 120),
    "ascent":    (0,   255,  80),
    "apex":      (0,   220, 255),
    "descent":   (0,   100, 255),
    "completed": (200,   0, 255),
}

_SHOT_COLOR = {
    "GOOD SHOT": (0,   255,   0),
    "MISS":      (0,    60, 255),
    "IN FLIGHT": (0,   220, 255),
    "TRACKING":  (180, 180, 180),
}


def _draw_trail(frame: np.ndarray, trail: list, tracker) -> None:
    """Draw the colour-coded trajectory trail."""
    history = list(tracker._history)
    n_draw  = min(len(history), TRAIL_MAX_POINTS)
    recent  = history[-n_draw:]

    for i in range(1, len(recent)):
        phase = recent[i]["phase"]
        color = _PHASE_COLOR.get(phase, (180, 180, 180))
        alpha = 0.4 + 0.6 * (i / len(recent))    # fade older points
        color = tuple(int(c * alpha) for c in color)
        pt1 = recent[i - 1]["pos"]
        pt2 = recent[i]["pos"]
        thickness = 1 + int(2 * (i / len(recent)))
        cv2.line(frame, pt1, pt2, color, thickness, cv2.LINE_AA)


def _draw_arc(frame: np.ndarray, arc_points: list) -> None:
    """Draw the predicted parabolic arc (dashed line)."""
    for i in range(1, len(arc_points)):
        if i % 2 == 0:         # draw every-other segment for dashed effect
            continue
        pt1 = (int(arc_points[i-1][0]), int(arc_points[i-1][1]))
        pt2 = (int(arc_points[i][0]),   int(arc_points[i][1]))
        cv2.line(frame, pt1, pt2, (255, 200, 0), 2, cv2.LINE_AA)


def _draw_ball(frame: np.ndarray, detection: dict) -> None:
    """Draw the detected ball circle + source label."""
    center = detection.get("center")
    radius = detection.get("radius")
    if center is None or radius is None:
        return

    src   = detection.get("source", "")
    color = (0, 255, 80) if "yolo" in src else (0, 200, 255)
    cv2.circle(frame, center, max(1, int(radius)), color, 2, cv2.LINE_AA)
    cv2.circle(frame, center, 3, color, -1, cv2.LINE_AA)

    label = f"{src} {detection.get('confidence', 0):.2f}" if detection.get("confidence") else src
    cv2.putText(frame, label,
                (center[0] + int(radius) + 4, center[1] - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)


def _draw_rim(frame: np.ndarray, rim: dict) -> None:
    """Draw the rim circle (or an absent indicator)."""
    if rim.get("detected"):
        cv2.circle(frame, (rim["x"], rim["y"]), rim["radius"], (255, 80, 255), 2, cv2.LINE_AA)
        cv2.circle(frame, (rim["x"], rim["y"]), 4, (255, 80, 255), -1, cv2.LINE_AA)
    else:
        cv2.putText(frame, "Rim: Not In View",
                    (10, DISPLAY_HEIGHT - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 60, 255), 2, cv2.LINE_AA)


def _draw_hud(frame: np.ndarray, shot, tracker, detection: dict) -> None:
    """Render the analytics HUD panel in the top-left corner."""
    phase          = tracker.phase
    release_angle  = shot.release_angle
    entry_angle    = shot.entry_angle
    apex           = shot.apex_height
    prediction     = shot.shot_prediction
    src            = detection.get("source", "none")

    panel_color   = (20, 20, 20)
    text_color    = (220, 220, 220)
    accent        = _SHOT_COLOR.get(prediction, (200, 200, 200))

    # Semi-transparent panel background
    overlay = frame.copy()
    cv2.rectangle(overlay, (5, 5), (260, 165), panel_color, -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    lines = [
        (f"Phase:   {phase.upper()}",                   text_color),
        (f"Detector: {src}",                            text_color),
        (f"Release: {shot.angle_str}",                  (100, 255, 150)),
        (f"Entry:   {'—' if entry_angle is None else f'{entry_angle:.1f}°'}",
                                                         (100, 200, 255)),
        (f"Apex Δy: {'—' if apex is None else f'{apex:.0f}px'}",
                                                         (255, 200, 60)),
        (f"Shot:    {prediction}",                       accent),
    ]

    for idx, (text, color) in enumerate(lines):
        cv2.putText(frame, text,
                    (12, 30 + idx * 23),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.58, color, 1, cv2.LINE_AA)


# ──────────────────────────────────────────────────────────────────────────────
# Main loop
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    if not os.path.isfile(args.video):
        print(f"[ERROR] Video not found: {args.video}")
        sys.exit(1)

    # ── Open video ────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video: {args.video}")
        sys.exit(1)

    video_name = os.path.splitext(os.path.basename(args.video))[0]
    native_w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    native_h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps        = cap.get(cv2.CAP_PROP_FPS) or 30.0
    print(f"[MAIN] Video: {video_name}  {native_w}×{native_h} @ {fps:.1f} fps")

    # ── Rim detection (multi-frame) ───────────────────────────────────────
    print("[MAIN] Detecting rim …")
    rim = get_rim_position(cap, debug=True)
    print(f"[MAIN] Rim result: {rim}")

    # ── Initialise pipeline modules ───────────────────────────────────────
    detector = BallDetector(mode=args.detector)
    tracker  = BallTracker()
    analyser = ShotAnalyser(rim)

    # ── Video writer (optional) ───────────────────────────────────────────
    writer = None
    if args.save_video:
        os.makedirs("output", exist_ok=True)
        out_path = os.path.join("output", f"annotated_{video_name}.mp4")
        fourcc   = cv2.VideoWriter_fourcc(*"mp4v")
        writer   = cv2.VideoWriter(out_path, fourcc, fps,
                                   (DISPLAY_WIDTH, DISPLAY_HEIGHT))
        print(f"[MAIN] Saving annotated video → {out_path}")

    # ── Per-shot output accumulator ───────────────────────────────────────
    os.makedirs("output", exist_ok=True)
    shot_log: list[dict] = []
    prev_phase = "held"

    # ── Main processing loop ──────────────────────────────────────────────
    frame_idx = 0
    while True:
        ret, raw_frame = cap.read()
        if not ret:
            break

        frame = letterbox(raw_frame, DISPLAY_WIDTH, DISPLAY_HEIGHT)

        # Detection → Tracking → Analysis
        detection = detector.detect(frame)
        tracker.update(detection)
        shot = analyser.update(tracker)

        # Log shot completion when phase transitions to held after a flight
        if prev_phase in ("descent", "apex") and tracker.phase == "held":
            if shot.arc_points:
                shot_log.append({
                    "frame":          frame_idx,
                    "release_angle":  shot.release_angle,
                    "entry_angle":    shot.entry_angle,
                    "apex_height_px": shot.apex_height,
                    "prediction":     shot.shot_prediction,
                    "arc_points":     shot.arc_points,
                    "arc_coeffs_x":   shot.arc_coeffs_x,
                    "arc_coeffs_y":   shot.arc_coeffs_y,
                    "rim_detected":   shot.rim_detected,
                })
        prev_phase = tracker.phase

        # ── Draw everything ───────────────────────────────────────────────
        _draw_trail(frame, tracker.trail, tracker)
        _draw_arc(frame, shot.arc_points)
        _draw_ball(frame, detection)
        _draw_rim(frame, rim)
        _draw_hud(frame, shot, tracker, detection)

        # ── Display & write ───────────────────────────────────────────────
        cv2.imshow("Smart Basketball Tracker", frame)
        if writer is not None:
            writer.write(frame)

        frame_idx += 1

        # ESC or 'q' to quit
        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord("q")):
            print("[MAIN] Stopped by user.")
            break

    # ── Cleanup ───────────────────────────────────────────────────────────
    cap.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()

    # ── Save trajectory JSON ──────────────────────────────────────────────
    # Save final tracker history as well as per-shot log
    trajectory_list = tracker.trail
    output_data = {
        "video":      os.path.basename(args.video),
        "total_frames": frame_idx,
        "rim":        rim,
        "trajectory": [[int(p[0]), int(p[1])] for p in trajectory_list],
        "shots":      shot_log,
    }
    json_path = os.path.join("output", "trajectory_data.json")
    with open(json_path, "w") as f:
        json.dump(output_data, f, indent=2)
    print(f"[MAIN] Trajectory data saved → {json_path}")
    print(f"[MAIN] Shots logged: {len(shot_log)}")


if __name__ == "__main__":
    main()