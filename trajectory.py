"""
trajectory.py
=============
BallTracker – a lightweight 2-D Kalman filter tracker for a single basketball.

State vector:  x = [px, py, vx, vy]  (position + velocity in pixels / frame)
Measurement:   z = [px, py]           (detected ball centre)

Key features
------------
* Measurement gating  – rejects detections that jump further than
  MAX_BALL_TELEPORT_DISTANCE pixels in one step.
* Occlusion coasting  – predicts ball position for up to MAX_COAST_FRAMES
  consecutive frames without a valid measurement.
* Flight-segment extraction – separates the shot flight arc from
  dribble / held trajectory points so the parabolic fit is applied
  only to free-flight data.
* History rolling window – stores up to MAX_TRAJECTORY_POINTS timestamped
  points with velocity and phase tags.
"""

import cv2
import numpy as np
from collections import deque

from config import (
    MAX_BALL_TELEPORT_DISTANCE,
    MAX_COAST_FRAMES,
    MAX_TRAJECTORY_POINTS,
)

# ── Kalman filter dimensions ──────────────────────────────────────────────────
_MEAS_DIM  = 2   # [px, py]
_STATE_DIM = 4   # [px, py, vx, vy]


class BallTracker:
    """
    Single-object 2-D Kalman filter tracker for a basketball.

    Usage
    -----
        tracker = BallTracker()
        # Each frame:
        tracker.update(detection_result)   # dict from BallDetector.detect()
        pts  = tracker.trail               # list of (x,y) for drawing
        vel  = tracker.velocity            # (vx, vy) current estimate
        pos  = tracker.position            # (x, y)  current estimate or None
        seg  = tracker.flight_segment()    # free-flight (x,y) points only
    """

    def __init__(self) -> None:
        self._kf          = self._build_kalman()
        self._initialised = False
        self._coast_count = 0                          # consecutive frames without measurement
        self._history: deque[dict] = deque(maxlen=MAX_TRAJECTORY_POINTS)
        self._phase       = "held"                     # held | ascent | apex | descent | completed

    # ── Kalman setup ──────────────────────────────────────────────────────────

    @staticmethod
    def _build_kalman() -> cv2.KalmanFilter:
        kf = cv2.KalmanFilter(_STATE_DIM, _MEAS_DIM)

        # Transition matrix:  position += velocity each step
        kf.transitionMatrix = np.array(
            [[1, 0, 1, 0],
             [0, 1, 0, 1],
             [0, 0, 1, 0],
             [0, 0, 0, 1]], dtype=np.float32
        )

        # Measurement matrix: we observe [px, py]
        kf.measurementMatrix = np.array(
            [[1, 0, 0, 0],
             [0, 1, 0, 0]], dtype=np.float32
        )

        kf.processNoiseCov      = np.eye(_STATE_DIM, dtype=np.float32) * 1e-2
        kf.measurementNoiseCov  = np.eye(_MEAS_DIM,  dtype=np.float32) * 1e-1
        kf.errorCovPost         = np.eye(_STATE_DIM, dtype=np.float32)
        return kf

    # ── Public update method ──────────────────────────────────────────────────

    def update(self, detection: dict) -> None:
        """
        Feed one frame's detection result (from BallDetector.detect()).
        Call this every frame regardless of whether a detection was found.
        """
        center = detection.get("center")

        # ── Measurement gating ────────────────────────────────────────────
        if center is not None and self._initialised:
            prev = self.position
            if prev is not None:
                dist = np.hypot(center[0] - prev[0], center[1] - prev[1])
                if dist > MAX_BALL_TELEPORT_DISTANCE:
                    center = None   # reject – too far

        # ── Kalman predict ────────────────────────────────────────────────
        predicted = self._kf.predict()

        # ── Kalman correct ────────────────────────────────────────────────
        if center is not None:
            if not self._initialised:
                # Initialise state with first good detection
                self._kf.statePre[0, 0] = center[0]
                self._kf.statePre[1, 0] = center[1]
                self._kf.statePre[2, 0] = 0.0
                self._kf.statePre[3, 0] = 0.0
                self._initialised = True

            measurement = np.array([[center[0]], [center[1]]], dtype=np.float32)
            self._kf.correct(measurement)
            self._coast_count = 0
        else:
            self._coast_count += 1

        # ── Update phase & history ────────────────────────────────────────
        if self._initialised and self._coast_count <= MAX_COAST_FRAMES:
            px, py = int(predicted[0, 0]), int(predicted[1, 0])
            vx, vy = float(predicted[2, 0]), float(predicted[3, 0])
            self._update_phase(vy)
            self._history.append({
                "pos":         (px, py),
                "vel":         (vx, vy),
                "phase":       self._phase,
                "measured":    center is not None,
                "confidence":  detection.get("confidence"),
                "source":      detection.get("source", "none"),
            })

    def reset_shot(self) -> None:
        """Call this after a shot is completed to begin a fresh trajectory."""
        self._history.clear()
        self._phase = "held"

    # ── Phase state machine ───────────────────────────────────────────────────

    def _update_phase(self, vy: float) -> None:
        if self._phase == "completed":
            return
        if self._phase == "held":
            if vy < -1.5:                   # moving upwards
                self._phase = "ascent"
        elif self._phase == "ascent":
            if vy >= -0.5:                  # velocity ≈ 0 → apex
                self._phase = "apex"
        elif self._phase == "apex":
            if vy > 1.5:                    # now falling
                self._phase = "descent"

    # ── Convenience properties ────────────────────────────────────────────────

    @property
    def position(self) -> tuple[int, int] | None:
        if not self._initialised or self._coast_count > MAX_COAST_FRAMES:
            return None
        return self._history[-1]["pos"] if self._history else None

    @property
    def velocity(self) -> tuple[float, float]:
        if not self._history:
            return (0.0, 0.0)
        return self._history[-1]["vel"]

    @property
    def phase(self) -> str:
        return self._phase

    @property
    def trail(self) -> list[tuple[int, int]]:
        """Smoothed position history as (x, y) tuples."""
        return [h["pos"] for h in self._history]

    def flight_segment(self) -> list[dict]:
        """Return only frames where phase is ascent / apex / descent."""
        return [h for h in self._history
                if h["phase"] in ("ascent", "apex", "descent")]