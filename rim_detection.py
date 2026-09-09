"""
rim_detection.py
================
Robust basketball rim (hoop) detector using a multi-frame accumulation strategy.

Algorithm
---------
1. Sample up to RIM_DETECTION_FRAMES evenly spaced frames from the video.
2. In each frame apply an orange HSV mask (the rim is always orange/red metal)
   combined with Canny edges.
3. Run HoughCircles on the combined edge/colour map.
4. Accumulate all candidates; vote by clustering nearby candidates (DBSCAN-style).
5. Select the most-voted cluster.  If confidence < RIM_CONFIDENCE_THRESH,
   return detected=False (e.g. outdoor driveways with no hoop visible).
6. Optionally fit an ellipse to handle perspective projection from side cameras.

Fallback
--------
If detection fails, return the config fallback values with detected=False so
callers can gate shot prediction appropriately.
"""

import cv2
import numpy as np
import os

from config import (
    DISPLAY_WIDTH, DISPLAY_HEIGHT,
    RIM_DETECTION_FRAMES, RIM_CONFIDENCE_THRESH,
    RIM_X, RIM_Y, RIM_RADIUS,
)


# HSV range for the orange rim (slightly wider than ball range to capture metal)
_RIM_LOWER = np.array([3,  80, 60],  dtype=np.uint8)
_RIM_UPPER = np.array([25, 255, 255], dtype=np.uint8)


# ──────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────────────────────────────────────

def _sample_frames(cap: cv2.VideoCapture, n: int) -> list[np.ndarray]:
    """Return up to n evenly-spaced frames from a capture, leaving cap at pos 0."""
    total    = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    indices  = np.linspace(0, max(0, total - 1), min(n, total), dtype=int)
    frames   = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if ret and frame is not None:
            frames.append(cv2.resize(frame, (DISPLAY_WIDTH, DISPLAY_HEIGHT)))
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    return frames


def _candidate_circles_in_frame(frame: np.ndarray) -> list[tuple[int, int, int]]:
    """Detect hoop-like circles in one frame. Returns list of (cx, cy, cr)."""
    h, w = frame.shape[:2]

    blurred = cv2.GaussianBlur(frame, (5, 5), 0)
    hsv     = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    # Orange mask – isolate rim metal colour
    orange_mask = cv2.inRange(hsv, _RIM_LOWER, _RIM_UPPER)

    # Also include white mask for the net backboard area (helps Hough stability)
    white_mask = cv2.inRange(
        hsv, np.array([0, 0, 160]), np.array([180, 50, 255])
    )
    combined_mask = cv2.bitwise_or(orange_mask, white_mask)

    # Edge detection applied to the mask region
    gray  = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 40, 120)
    edges = cv2.bitwise_and(edges, combined_mask)

    # Search only in upper 70 % of frame (rim is above midcourt)
    search_edges = edges.copy()
    search_edges[int(h * 0.70):, :] = 0

    circles = cv2.HoughCircles(
        search_edges,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=30,
        param1=80,
        param2=15,
        minRadius=10,
        maxRadius=65,
    )

    if circles is None:
        return []

    results = []
    circles = np.round(circles[0]).astype(int)
    for cx, cy, cr in circles:
        # Exclude bottom 30 % of frame
        if cy > h * 0.70:
            continue
        # Additional sanity – rim radius ≈ 10–65 px at 832×464
        if cr < 10 or cr > 65:
            continue
        results.append((int(cx), int(cy), int(cr)))
    return results


def _cluster_candidates(
    candidates: list[tuple[int, int, int]],
    eps: float = 40.0,
) -> tuple[int, int, int] | None:
    """
    Simple single-pass density clustering: group candidates within `eps` pixels,
    pick the group with the most votes, return its centroid.
    """
    if not candidates:
        return None

    used    = [False] * len(candidates)
    clusters: list[list[int]] = []

    for i, (cx, cy, _) in enumerate(candidates):
        if used[i]:
            continue
        group = [i]
        used[i] = True
        for j, (ox, oy, _) in enumerate(candidates):
            if not used[j] and np.hypot(cx - ox, cy - oy) < eps:
                group.append(j)
                used[j] = True
        clusters.append(group)

    best_group = max(clusters, key=len)
    gx   = int(np.mean([candidates[i][0] for i in best_group]))
    gy   = int(np.mean([candidates[i][1] for i in best_group]))
    gr   = int(np.mean([candidates[i][2] for i in best_group]))
    return gx, gy, gr


# ──────────────────────────────────────────────────────────────────────────────
# Public function
# ──────────────────────────────────────────────────────────────────────────────

def get_rim_position(cap: cv2.VideoCapture | None = None, debug: bool = True) -> dict:
    """
    Detect the basketball rim from a video stream.

    Returns
    -------
    dict with keys:
        x, y, radius  – rim centre and radius in display pixels
        detected      – True if confidently found, False if fallback / absent
        confidence    – fraction of sampled frames that contained a candidate
        reason        – human-readable status string
    """
    fallback = {
        "x":          RIM_X,
        "y":          RIM_Y,
        "radius":     RIM_RADIUS,
        "detected":   False,
        "confidence": 0.0,
        "reason":     "No video provided – using config defaults",
    }

    if cap is None:
        return fallback

    # ── Sample frames ─────────────────────────────────────────────────────
    frames = _sample_frames(cap, RIM_DETECTION_FRAMES)
    if not frames:
        fallback["reason"] = "Failed to read frames"
        return fallback

    # ── Collect candidates across all sampled frames ───────────────────────
    all_candidates: list[tuple[int, int, int]] = []
    frames_with_hit = 0

    for frame in frames:
        hits = _candidate_circles_in_frame(frame)
        if hits:
            frames_with_hit += 1
            all_candidates.extend(hits)

    confidence = frames_with_hit / len(frames) if frames else 0.0

    print(f"[RIM] Sampled {len(frames)} frames, "
          f"{frames_with_hit} had candidates ({confidence * 100:.0f}% confidence). "
          f"Total raw candidates: {len(all_candidates)}")

    # ── Confidence gate ───────────────────────────────────────────────────
    if confidence < RIM_CONFIDENCE_THRESH or not all_candidates:
        fallback["detected"]   = False
        fallback["confidence"] = confidence
        fallback["reason"]     = (
            "Rim not confidently visible in video "
            f"(conf={confidence:.2f} < threshold={RIM_CONFIDENCE_THRESH})"
        )
        print(f"[RIM] {fallback['reason']}")
        return fallback

    # ── Cluster & pick best ───────────────────────────────────────────────
    result = _cluster_candidates(all_candidates, eps=50.0)
    if result is None:
        fallback["reason"] = "Clustering produced no consensus"
        return fallback

    rx, ry, rr = result
    print(f"[RIM] Detected at ({rx}, {ry}), r={rr}, confidence={confidence:.2f}")

    if debug:
        os.makedirs("output", exist_ok=True)
        dbg = frames[len(frames) // 2].copy()
        cv2.circle(dbg, (rx, ry), rr,   (0, 255, 0),   3)
        cv2.circle(dbg, (rx, ry), 4,    (0, 0, 255),  -1)
        cv2.putText(dbg, f"Rim conf={confidence:.2f}", (rx - 60, ry - rr - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imwrite("output/debug_detected_rim.png", dbg)

    return {
        "x":          rx,
        "y":          ry,
        "radius":     rr,
        "detected":   True,
        "confidence": confidence,
        "reason":     "OK",
    }