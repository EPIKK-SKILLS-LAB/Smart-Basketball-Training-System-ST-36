"""Headless benchmark: measure detection rates on both test videos."""
import cv2, os, sys, numpy as np
sys.path.insert(0, os.path.dirname(__file__))

from config import DISPLAY_WIDTH, DISPLAY_HEIGHT
from ball_detection import BallDetector
from trajectory import BallTracker
from rim_detection import get_rim_position


def letterbox(frame, w=DISPLAY_WIDTH, h=DISPLAY_HEIGHT):
    h0, w0 = frame.shape[:2]
    scale = min(w / w0, h / h0)
    nw, nh = int(w0 * scale), int(h0 * scale)
    res = cv2.resize(frame, (nw, nh))
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    yo = (h - nh) // 2
    xo = (w - nw) // 2
    canvas[yo:yo+nh, xo:xo+nw] = res
    return canvas


detector = BallDetector(mode="yolo")

for vid in ["resrc/testSub1.mp4", "resrc/testSub2.mp4"]:
    print(f"\n{'='*55}")
    print(f"  {os.path.basename(vid)}")
    print(f"{'='*55}")

    cap = cv2.VideoCapture(vid)
    rim = get_rim_position(cap, debug=False)
    print(f"Rim detected : {rim['detected']}  conf={rim['confidence']:.2f}")
    print(f"Rim position : ({rim['x']}, {rim['y']})  r={rim['radius']}")
    print(f"Rim reason   : {rim['reason']}")

    tracker = BallTracker()
    total = yolo_hits = classical_hits = no_det = 0
    radii = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = letterbox(frame)
        det = detector.detect(frame)
        tracker.update(det)
        src = det.get("source", "none")
        if "yolo" in src:
            yolo_hits += 1
            radii.append(det["radius"])
        elif "classical" in src:
            classical_hits += 1
            radii.append(det["radius"])
        else:
            no_det += 1
        total += 1

    cap.release()
    print(f"\nFrames total       : {total}")
    print(f"YOLO detections    : {yolo_hits:4d}  ({yolo_hits/total*100:.1f}%)")
    print(f"Classical fallback : {classical_hits:4d}  ({classical_hits/total*100:.1f}%)")
    print(f"No detection       : {no_det:4d}  ({no_det/total*100:.1f}%)")
    if radii:
        print(f"Radius  min/max/avg: {min(radii):.1f} / {max(radii):.1f} / {np.mean(radii):.1f} px")
