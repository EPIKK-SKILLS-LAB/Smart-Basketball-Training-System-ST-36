# ──────────────────────────────────────────────────────────────────────────────
# DISPLAY / EXPORT SETTINGS
# Preserve the native 832×464 (16:9) aspect ratio in the display window.
# YOLO inference will letterbox internally to 640×640 on its own.
# ──────────────────────────────────────────────────────────────────────────────
DISPLAY_WIDTH  = 832
DISPLAY_HEIGHT = 464

# ──────────────────────────────────────────────────────────────────────────────
# DETECTOR SETTINGS
# ──────────────────────────────────────────────────────────────────────────────
# "yolo"      → YOLOv8-nano primary (auto-downloaded first run) + classical fallback
# "classical" → OpenCV-only MOG2 + colour + shape pipeline
DETECTOR_TYPE = "yolo"

MODEL_PATH          = "yolov8n.pt"   # Downloaded to cwd on first run
CONF_THRESHOLD_BALL = 0.20           # Min YOLO confidence for class=sports ball
CONF_THRESHOLD_PERSON = 0.40         # Min YOLO confidence for class=person

# COCO class IDs used by YOLOv8n
COCO_BALL_CLASS   = 32   # sports ball
COCO_PERSON_CLASS = 0    # person

# ──────────────────────────────────────────────────────────────────────────────
# CLASSICAL FALLBACK – Ball HSV Ranges
# Two ranges: standard orange + deep red/maroon outdoor balls
# ──────────────────────────────────────────────────────────────────────────────
BALL_HSV_RANGES = [
    ([5,  80, 60],  [28, 255, 255]),   # Orange (NBA, standard)
    ([0,  50, 40],  [10, 255, 220]),   # Dark red/maroon (outdoor composite)
    ([165, 50, 40], [180, 255, 220]),  # Dark red wrap-around
]
MIN_CIRCULARITY  = 0.55
MIN_BALL_RADIUS  = 4       # px (640×480 equivalent)
MAX_BALL_RADIUS  = 45      # px
MIN_BALL_AREA    = 50      # px²
MAX_BALL_AREA    = 6500    # px²

# ──────────────────────────────────────────────────────────────────────────────
# KALMAN / TRACKING SETTINGS
# ──────────────────────────────────────────────────────────────────────────────
MAX_BALL_TELEPORT_DISTANCE = 150   # px – measurements further than this are rejected
MAX_COAST_FRAMES           = 6     # frames the tracker predicts without a measurement
MAX_TRAJECTORY_POINTS      = 120   # total trail history

# ──────────────────────────────────────────────────────────────────────────────
# RIM DETECTION SETTINGS
# ──────────────────────────────────────────────────────────────────────────────
RIM_DETECTION_FRAMES   = 60    # frames sampled for multi-frame accumulation
RIM_CONFIDENCE_THRESH  = 0.35  # below this → "Rim Not In View"

# Fallback values (used only if auto-detection completely fails)
RIM_X      = 306
RIM_Y      = 118
RIM_RADIUS = 48

# ──────────────────────────────────────────────────────────────────────────────
# SHOT ANALYSIS SETTINGS
# ──────────────────────────────────────────────────────────────────────────────
SHOT_MIN_FLIGHT_FRAMES = 8     # minimum frames of upward flight to classify a shot
SHOT_MIN_UPWARD_PX     = 15    # minimum vertical rise (px) to classify release

# ──────────────────────────────────────────────────────────────────────────────
# TRAJECTORY VISUALISATION
# ──────────────────────────────────────────────────────────────────────────────
TRAIL_MAX_POINTS = 60          # points shown in the on-screen trail