"""
ball_detection.py
=================
Dual-mode ball detector:
  Primary  – YOLOv8-nano via the `ultralytics` library (class 32 = sports ball).
  Fallback – Classical OpenCV pipeline: MOG2 motion segmentation + multi-range
             adaptive HSV colour filter + strict circularity / aspect-ratio shape
             validation.  Activated automatically when ultralytics is unavailable
             or when DETECTOR_TYPE != "yolo".

Public API
----------
    detector = BallDetector(mode="yolo")   # or "classical"
    result   = detector.detect(frame)
    # result: {"center": (x,y)|None, "radius": float|None,
    #          "confidence": float|None, "mask": np.ndarray, "source": str}
"""

import cv2
import numpy as np

from config import (
    DETECTOR_TYPE,
    MODEL_PATH,
    CONF_THRESHOLD_BALL,
    COCO_BALL_CLASS,
    BALL_HSV_RANGES,
    MIN_CIRCULARITY,
    MIN_BALL_RADIUS,
    MAX_BALL_RADIUS,
    MIN_BALL_AREA,
    MAX_BALL_AREA,
)


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _circularity(cnt) -> float:
    area      = cv2.contourArea(cnt)
    perimeter = cv2.arcLength(cnt, True)
    if perimeter == 0:
        return 0.0
    return (4.0 * np.pi * area) / (perimeter ** 2)


def _empty_result(mask: np.ndarray) -> dict:
    return {
        "center":     None,
        "radius":     None,
        "confidence": None,
        "mask":       mask,
        "source":     "none",
    }


# ──────────────────────────────────────────────────────────────────────────────
# Classical pipeline
# ──────────────────────────────────────────────────────────────────────────────

class _ClassicalDetector:
    """
    MOG2 background subtraction + multi-range HSV colour filter +
    circularity / aspect-ratio / size geometry filter.
    """

    def __init__(self) -> None:
        self._bg = cv2.createBackgroundSubtractorMOG2(
            history=200, varThreshold=30, detectShadows=False
        )
        self._morph_kernel = np.ones((5, 5), np.uint8)

    def detect(self, frame: np.ndarray) -> dict:
        h, w = frame.shape[:2]
        blurred = cv2.GaussianBlur(frame, (5, 5), 0)
        hsv     = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

        # ── Colour mask (multi-range) ──────────────────────────────────────
        colour_mask = np.zeros((h, w), dtype=np.uint8)
        for lo, hi in BALL_HSV_RANGES:
            colour_mask = cv2.bitwise_or(
                colour_mask,
                cv2.inRange(hsv, np.array(lo), np.array(hi)),
            )

        # ── Motion mask ───────────────────────────────────────────────────
        motion_mask = self._bg.apply(blurred)
        motion_mask = cv2.morphologyEx(
            motion_mask, cv2.MORPH_OPEN, self._morph_kernel
        )

        # Combine: keep colour pixels that are also moving
        combined = cv2.bitwise_and(colour_mask, motion_mask)
        combined = cv2.morphologyEx(
            combined, cv2.MORPH_CLOSE, self._morph_kernel
        )
        combined = cv2.morphologyEx(
            combined, cv2.MORPH_OPEN, self._morph_kernel
        )

        # ── Contour filtering ─────────────────────────────────────────────
        contours, _ = cv2.findContours(
            combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        best        = None
        best_score  = -1.0

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < MIN_BALL_AREA or area > MAX_BALL_AREA:
                continue

            circ = _circularity(cnt)
            if circ < MIN_CIRCULARITY:
                continue

            (cx, cy), radius = cv2.minEnclosingCircle(cnt)
            if radius < MIN_BALL_RADIUS or radius > MAX_BALL_RADIUS:
                continue

            x_b, y_b, w_b, h_b = cv2.boundingRect(cnt)
            aspect = float(w_b) / h_b if h_b > 0 else 0
            if aspect < 0.65 or aspect > 1.55:
                continue

            # Score = circularity (0–1) as primary signal
            score = circ
            if score > best_score:
                best_score = score
                best = {"center": (int(cx), int(cy)),
                        "radius": float(radius),
                        "confidence": round(circ, 3)}

        if best is None:
            return _empty_result(combined)

        return {**best, "mask": combined, "source": "classical"}


# ──────────────────────────────────────────────────────────────────────────────
# YOLOv8 primary detector
# ──────────────────────────────────────────────────────────────────────────────

class _YOLODetector:
    """
    YOLOv8-nano sports-ball detector (class 32).  Model weights are downloaded
    automatically to the working directory on first run (~6 MB).
    Falls back to _ClassicalDetector if the model cannot be loaded.
    """

    def __init__(self) -> None:
        self._model     = None
        self._fallback  = _ClassicalDetector()
        self._load_model()

    def _load_model(self) -> None:
        try:
            from ultralytics import YOLO
            self._model = YOLO(MODEL_PATH)
            print(f"[BallDetector] YOLOv8 model loaded: {MODEL_PATH}")
        except Exception as exc:
            print(f"[BallDetector] YOLO load failed ({exc}). Using classical fallback.")
            self._model = None

    def detect(self, frame: np.ndarray) -> dict:
        # Build a blank mask for YOLO path (no colour mask needed)
        h, w  = frame.shape[:2]
        blank = np.zeros((h, w), dtype=np.uint8)

        if self._model is None:
            return self._fallback.detect(frame)

        try:
            results = self._model.predict(
                frame,
                conf=CONF_THRESHOLD_BALL,
                classes=[COCO_BALL_CLASS],
                verbose=False,
            )
        except Exception as exc:
            print(f"[BallDetector] YOLO inference error: {exc}")
            return self._fallback.detect(frame)

        best     = None
        best_conf = -1.0

        for r in results:
            if r.boxes is None:
                continue
            for box in r.boxes:
                cls  = int(box.cls[0])
                conf = float(box.conf[0])
                if cls != COCO_BALL_CLASS:
                    continue
                if conf < CONF_THRESHOLD_BALL:
                    continue

                x1, y1, x2, y2 = [float(v) for v in box.xyxy[0]]
                cx     = (x1 + x2) / 2.0
                cy     = (y1 + y2) / 2.0
                radius = max((x2 - x1), (y2 - y1)) / 2.0

                if conf > best_conf:
                    best_conf = conf
                    best = {
                        "center":     (int(cx), int(cy)),
                        "radius":     radius,
                        "confidence": round(conf, 3),
                        "mask":       blank,
                        "source":     "yolo",
                    }

        if best is None:
            # YOLO found nothing – try classical pipeline for extra robustness
            classical = self._fallback.detect(frame)
            if classical["center"] is not None:
                classical["source"] = "classical_fallback"
            return classical

        # Draw ball region on the mask for visualisation consistency
        cv2.circle(blank, best["center"], int(best["radius"]), 255, -1)
        best["mask"] = blank
        return best


# ──────────────────────────────────────────────────────────────────────────────
# Public factory
# ──────────────────────────────────────────────────────────────────────────────

class BallDetector:
    """
    Usage::

        detector = BallDetector()          # mode chosen by config.DETECTOR_TYPE
        result   = detector.detect(frame)
    """

    def __init__(self, mode: str | None = None) -> None:
        chosen = (mode or DETECTOR_TYPE).lower()
        if chosen == "yolo":
            self._impl = _YOLODetector()
        else:
            self._impl = _ClassicalDetector()
        self.mode = chosen

    def detect(self, frame: np.ndarray) -> dict:
        """
        Returns
        -------
        dict with keys:
            center      : (int x, int y) | None
            radius      : float | None
            confidence  : float | None
            mask        : np.ndarray (H×W uint8)
            source      : "yolo" | "classical" | "classical_fallback" | "none"
        """
        return self._impl.detect(frame)