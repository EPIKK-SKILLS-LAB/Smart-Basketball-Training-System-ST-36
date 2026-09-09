"""
shot_analysis.py
================
Physics-based shot analysis operating on BallTracker flight segments.

Public API
----------
    analyser = ShotAnalyser(rim)      # rim dict from get_rim_position()
    result   = analyser.update(tracker)  # call every frame
    # result: ShotResult dataclass or None if no shot in progress
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from config import SHOT_MIN_FLIGHT_FRAMES, SHOT_MIN_UPWARD_PX


# ──────────────────────────────────────────────────────────────────────────────
# Data structures
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class ShotResult:
    phase:          str                 # current phase of the shot
    release_angle:  Optional[float]     # degrees above horizontal at release
    entry_angle:    Optional[float]     # degrees of descent at rim height
    apex_height:    Optional[float]     # pixels above release point
    arc_coeffs_x:   Optional[list]      # polynomial x(t) coefficients
    arc_coeffs_y:   Optional[list]      # polynomial y(t) coefficients
    arc_points:     list[list[int]]     # sampled (x, y) points on fitted arc
    shot_prediction: str                # "GOOD SHOT" | "MISS" | "IN FLIGHT" | "TRACKING"
    rim_detected:   bool

    @property
    def angle_str(self) -> str:
        if self.release_angle is None:
            return "—"
        return f"{self.release_angle:.1f}°"


# ──────────────────────────────────────────────────────────────────────────────
# Helper utilities
# ──────────────────────────────────────────────────────────────────────────────

def _angle_from_velocity(vx: float, vy: float) -> Optional[float]:
    """Launch/descent angle in degrees above horizontal. vy is in screen coords
    (positive = downward), so we negate it for a physics-convention angle."""
    if abs(vx) < 0.001 and abs(vy) < 0.001:
        return None
    return math.degrees(math.atan2(-vy, vx))


def _fit_arc(pts: list[dict]) -> tuple[
        Optional[list], Optional[list], list[list[int]]]:
    """
    Fit quadratic polynomials x(t) and y(t) to the flight segment.
    Returns (coeffs_x, coeffs_y, sampled_arc_points).
    """
    if len(pts) < 3:
        return None, None, []

    t  = np.arange(len(pts), dtype=float)
    xs = np.array([p["pos"][0] for p in pts], dtype=float)
    ys = np.array([p["pos"][1] for p in pts], dtype=float)

    try:
        cx = np.polyfit(t, xs, 2)
        cy = np.polyfit(t, ys, 2)
    except np.linalg.LinAlgError:
        return None, None, []

    sample_t  = np.linspace(0, len(pts) - 1, 30)
    arc_pts   = [
        [int(np.polyval(cx, ti)), int(np.polyval(cy, ti))]
        for ti in sample_t
    ]
    return cx.tolist(), cy.tolist(), arc_pts


def _predict_outcome(
    arc_pts: list[list[int]],
    rim: dict,
) -> str:
    """Check if the fitted arc passes through the rim cylinder."""
    if not arc_pts or not rim.get("detected", False):
        return "IN FLIGHT"

    rx = rim["x"]
    ry = rim["y"]
    rr = rim["radius"]
    tol = rr * 1.1          # ±10 % tolerance on the rim radius

    for pt in arc_pts:
        px, py = pt
        # ball descending (py > ry means it's below the rim plane — close)
        dist = math.hypot(px - rx, py - ry)
        if dist <= tol and py >= ry - rr * 0.5:
            return "GOOD SHOT"
    return "MISS"


# ──────────────────────────────────────────────────────────────────────────────
# Main analyser class
# ──────────────────────────────────────────────────────────────────────────────

class ShotAnalyser:
    """
    Stateful shot analyser.  Call ``update(tracker)`` every frame.

    Parameters
    ----------
    rim : dict  – output of ``get_rim_position()``.
    """

    def __init__(self, rim: dict) -> None:
        self._rim           = rim
        self._release_angle: Optional[float] = None
        self._entry_angle:   Optional[float] = None
        self._apex_y:        Optional[int]   = None   # screen y of apex (smallest y)
        self._release_y:     Optional[int]   = None

    def update(self, tracker) -> ShotResult:
        """
        Compute shot analytics from the current tracker state.

        Parameters
        ----------
        tracker : BallTracker

        Returns
        -------
        ShotResult
        """
        phase   = tracker.phase
        trail   = tracker.trail
        vx, vy  = tracker.velocity
        flight  = tracker.flight_segment()

        # ── Release angle (captured once when ascent begins) ──────────────
        if phase == "ascent" and self._release_angle is None:
            self._release_angle = _angle_from_velocity(vx, vy)
            if trail:
                self._release_y = trail[-1][1]

        # ── Track apex ────────────────────────────────────────────────────
        if phase in ("ascent", "apex") and trail:
            cur_y = trail[-1][1]
            if self._apex_y is None or cur_y < self._apex_y:
                self._apex_y = cur_y

        # ── Entry angle (captured when descent begins) ─────────────────────
        if phase == "descent" and self._entry_angle is None:
            self._entry_angle = _angle_from_velocity(vx, vy)

        # ── Arc fitting (only on enough flight points) ─────────────────────
        coeffs_x, coeffs_y, arc_pts = None, None, []
        if len(flight) >= SHOT_MIN_FLIGHT_FRAMES:
            coeffs_x, coeffs_y, arc_pts = _fit_arc(flight)

        # ── Shot outcome prediction ────────────────────────────────────────
        if phase in ("descent", "completed") and arc_pts:
            prediction = _predict_outcome(arc_pts, self._rim)
        elif phase in ("ascent", "apex") and len(flight) >= SHOT_MIN_FLIGHT_FRAMES:
            prediction = "IN FLIGHT"
        elif phase == "held":
            prediction = "TRACKING"
        else:
            prediction = "IN FLIGHT"

        # ── Apex height in pixels ─────────────────────────────────────────
        apex_height: Optional[float] = None
        if self._apex_y is not None and self._release_y is not None:
            apex_height = float(self._release_y - self._apex_y)   # positive = rose

        # ── Reset state when a new shot begins (held after completed) ──────
        if phase == "held":
            self._release_angle = None
            self._entry_angle   = None
            self._apex_y        = None
            self._release_y     = None

        return ShotResult(
            phase          = phase,
            release_angle  = self._release_angle,
            entry_angle    = self._entry_angle,
            apex_height    = apex_height,
            arc_coeffs_x   = coeffs_x,
            arc_coeffs_y   = coeffs_y,
            arc_points     = arc_pts,
            shot_prediction= prediction,
            rim_detected   = self._rim.get("detected", False),
        )