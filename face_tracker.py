"""
╔══════════════════════════════════════════════════════════════╗
║   Paralytic Patient Face Tracker                             ║
║   Head Movement + Eye Tracking + Blink Detection             ║
║   Algorithms: Kalman Filter + EAR + PnP Pose Estimation      ║
╚══════════════════════════════════════════════════════════════╝

Dependencies:
    pip install mediapipe opencv-python numpy

Controls:
    Q / ESC  → Quit
    R        → Reset calibration baseline
    S        → Save current session log to CSV
"""

import cv2
import mediapipe as mp
import numpy as np
import time
import csv
import os
from collections import deque
from dataclasses import dataclass, field
from typing import Optional, Tuple

# ─────────────────────────────────────────────
#  CONFIGURATION
# ─────────────────────────────────────────────
class Config:
    # Camera
    CAMERA_INDEX        = 0
    FRAME_WIDTH         = 1280
    FRAME_HEIGHT        = 720
    FPS_TARGET          = 30

    # Kalman filter noise params (lower = smoother, higher = more responsive)
    KALMAN_PROCESS_NOISE   = 1e-3
    KALMAN_MEASURE_NOISE   = 1e-1

    # Head movement thresholds (degrees)
    HEAD_THRESHOLD_YAW     = 8.0
    HEAD_THRESHOLD_PITCH   = 6.0
    HEAD_THRESHOLD_ROLL    = 5.0

    # Eye aspect ratio thresholds
    EAR_BLINK_THRESHOLD    = 0.20   # below this = closed
    EAR_OPEN_THRESHOLD     = 0.25   # above this = open  (hysteresis)
    BLINK_CONSEC_FRAMES    = 2      # min frames closed to count as blink
    BLINK_DOUBLE_MS        = 400    # ms window for double blink

    # Eye gaze thresholds (normalised iris offset, 0–1)
    IRIS_THRESHOLD_H       = 0.10   # left/right
    IRIS_THRESHOLD_V       = 0.08   # up/down

    # Smoothing window for EAR / iris (median filter)
    SMOOTH_WINDOW          = 5

    # Head pose smoothing (exponential)
    HEAD_EMA_ALPHA         = 0.25

    # Dwell-based command (seconds of stable gaze to trigger)
    DWELL_TIME_S           = 1.5

    # Log file
    LOG_FILE               = "face_tracker_log.csv"


# ─────────────────────────────────────────────
#  KALMAN FILTER  (1D, constant velocity model)
# ─────────────────────────────────────────────
class KalmanFilter1D:
    """Lightweight 1-D Kalman filter for smoothing scalar signals."""

    def __init__(self, process_noise: float = 1e-3, measure_noise: float = 1e-1):
        self.Q = process_noise   # process noise covariance
        self.R = measure_noise   # measurement noise covariance
        # State: [value, velocity]
        self.x = np.zeros((2, 1))
        self.P = np.eye(2) * 1.0
        self.F = np.array([[1, 1], [0, 1]], dtype=float)  # state transition
        self.H = np.array([[1, 0]], dtype=float)           # measurement matrix

    def update(self, z: float) -> float:
        # Predict
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + np.eye(2) * self.Q

        # Update
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T / S[0, 0]
        self.x += K * (z - (self.H @ self.x)[0, 0])
        self.P = (np.eye(2) - K @ self.H) @ self.P

        return float(self.x[0, 0])

    def reset(self, value: float = 0.0):
        self.x = np.array([[value], [0.0]])
        self.P = np.eye(2)


# ─────────────────────────────────────────────
#  MEDIAN BUFFER  (for EAR / iris)
# ─────────────────────────────────────────────
class MedianBuffer:
    def __init__(self, size: int = 5):
        self.buf = deque(maxlen=size)

    def update(self, v: float) -> float:
        self.buf.append(v)
        return float(np.median(self.buf))


# ─────────────────────────────────────────────
#  BLINK DETECTOR  (EAR + hysteresis + double)
# ─────────────────────────────────────────────
@dataclass
class BlinkState:
    is_closed:      bool  = False
    consec_closed:  int   = 0
    blink_count:    int   = 0
    last_blink_ms:  float = 0.0
    double_blink:   bool  = False


class BlinkDetector:
    def __init__(self):
        self.state = BlinkState()

    def update(self, ear: float, now_ms: float) -> BlinkState:
        s = self.state
        s.double_blink = False

        if ear < Config.EAR_BLINK_THRESHOLD:
            s.consec_closed += 1
        else:
            if s.is_closed and s.consec_closed >= Config.BLINK_CONSEC_FRAMES:
                # Blink completed
                s.blink_count += 1
                dt = now_ms - s.last_blink_ms
                if 0 < dt <= Config.BLINK_DOUBLE_MS:
                    s.double_blink = True
                s.last_blink_ms = now_ms
            s.consec_closed = 0

        s.is_closed = ear < Config.EAR_BLINK_THRESHOLD if not s.is_closed \
                      else ear < Config.EAR_OPEN_THRESHOLD   # hysteresis

        return s


# ─────────────────────────────────────────────
#  EYE ASPECT RATIO
# ─────────────────────────────────────────────
def eye_aspect_ratio(landmarks, indices: list, img_w: int, img_h: int) -> float:
    """
    EAR = (|P2-P6| + |P3-P5|) / (2 * |P1-P4|)
    Indices order: [corner_left, top1, top2, corner_right, bot2, bot1]
    """
    pts = np.array([
        [landmarks[i].x * img_w, landmarks[i].y * img_h]
        for i in indices
    ], dtype=float)

    A = np.linalg.norm(pts[1] - pts[5])
    B = np.linalg.norm(pts[2] - pts[4])
    C = np.linalg.norm(pts[0] - pts[3])
    return (A + B) / (2.0 * C + 1e-6)


# ─────────────────────────────────────────────
#  HEAD POSE (PnP + EMA smoothing)
# ─────────────────────────────────────────────
# Canonical 3-D face model (MediaPipe 468-landmark subset)
FACE_3D_MODEL = np.array([
    [0.0,    0.0,    0.0   ],   # Nose tip          (1)
    [0.0,   -330.0, -65.0  ],   # Chin              (152)
    [-225.0,  170.0,-135.0 ],   # Left eye corner   (263)
    [225.0,   170.0,-135.0 ],   # Right eye corner  (33)
    [-150.0, -150.0,-125.0 ],   # Left mouth corner (287)
    [150.0,  -150.0,-125.0 ],   # Right mouth corner(57)
], dtype=np.float64)

# Corresponding MediaPipe landmark indices
FACE_LANDMARK_IDS = [1, 152, 263, 33, 287, 57]


def get_head_pose(landmarks, img_w: int, img_h: int, cam_matrix: np.ndarray,
                  dist_coeffs: np.ndarray) -> Optional[Tuple[float, float, float]]:
    """Returns (yaw, pitch, roll) in degrees using solvePnP."""
    pts_2d = np.array([
        [landmarks[i].x * img_w, landmarks[i].y * img_h]
        for i in FACE_LANDMARK_IDS
    ], dtype=np.float64)

    success, rvec, tvec = cv2.solvePnP(
        FACE_3D_MODEL, pts_2d, cam_matrix, dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE
    )
    if not success:
        return None

    rmat, _ = cv2.Rodrigues(rvec)
    proj = np.hstack([rmat, tvec])
    _, _, _, _, _, _, euler = cv2.decomposeProjectionMatrix(proj)
    pitch = float(euler[0])
    yaw   = float(euler[1])
    roll  = float(euler[2])
    return yaw, pitch, roll


# ─────────────────────────────────────────────
#  IRIS GAZE (normalised offset within eye box)
# ─────────────────────────────────────────────
# MediaPipe iris landmark indices
LEFT_IRIS   = [474, 475, 476, 477]
RIGHT_IRIS  = [469, 470, 471, 472]
LEFT_EYE    = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
RIGHT_EYE   = [33,  7,   163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]

# Compact EAR indices (6-point)
LEFT_EAR_IDX  = [362, 385, 387, 263, 373, 380]
RIGHT_EAR_IDX = [33,  160, 158, 133, 153, 144]


def iris_gaze(landmarks, iris_ids: list, eye_ids: list,
              img_w: int, img_h: int) -> Tuple[float, float]:
    """
    Returns (h_offset, v_offset) in [-1, 1]:
        h_offset > 0 → looking right
        v_offset > 0 → looking down
    """
    iris_pts = np.array([[landmarks[i].x * img_w, landmarks[i].y * img_h]
                          for i in iris_ids])
    eye_pts  = np.array([[landmarks[i].x * img_w, landmarks[i].y * img_h]
                          for i in eye_ids])

    iris_cx = iris_pts[:, 0].mean()
    iris_cy = iris_pts[:, 1].mean()

    ex_min, ex_max = eye_pts[:, 0].min(), eye_pts[:, 0].max()
    ey_min, ey_max = eye_pts[:, 1].min(), eye_pts[:, 1].max()

    eye_cx = (ex_min + ex_max) / 2.0
    eye_cy = (ey_min + ey_max) / 2.0
    eye_hw = (ex_max - ex_min) / 2.0 + 1e-6
    eye_hh = (ey_max - ey_min) / 2.0 + 1e-6

    h = (iris_cx - eye_cx) / eye_hw   # [-1, 1]
    v = (iris_cy - eye_cy) / eye_hh   # [-1, 1]
    return float(np.clip(h, -1, 1)), float(np.clip(v, -1, 1))


# ─────────────────────────────────────────────
#  COMMAND INTERPRETER
# ─────────────────────────────────────────────
COMMANDS = {
    "head_left":    "← HEAD LEFT",
    "head_right":   "→ HEAD RIGHT",
    "head_up":      "↑ HEAD UP",
    "head_down":    "↓ HEAD DOWN",
    "eye_left":     "← GAZE LEFT",
    "eye_right":    "→ GAZE RIGHT",
    "eye_up":       "↑ GAZE UP",
    "eye_down":     "↓ GAZE DOWN",
    "blink":        "● BLINK",
    "double_blink": "●● DOUBLE BLINK",
    "neutral":      "○ NEUTRAL",
}

@dataclass
class DwellTracker:
    current_cmd: str   = "neutral"
    start_time:  float = 0.0
    confirmed:   bool  = False

    def update(self, cmd: str, now: float) -> bool:
        """Returns True when dwell time reached."""
        if cmd != self.current_cmd:
            self.current_cmd = cmd
            self.start_time  = now
            self.confirmed   = False
        elif not self.confirmed and (now - self.start_time) >= Config.DWELL_TIME_S:
            self.confirmed = True
            return True
        return False

    def progress(self, now: float) -> float:
        """0–1 progress toward dwell confirmation."""
        if self.confirmed:
            return 1.0
        return min(1.0, (now - self.start_time) / Config.DWELL_TIME_S)


# ─────────────────────────────────────────────
#  DISPLAY HELPERS
# ─────────────────────────────────────────────
COLORS = {
    "bg":       (15,  15,  20 ),
    "panel":    (28,  32,  40 ),
    "accent":   (0,   200, 160),
    "warning":  (0,   140, 255),
    "danger":   (0,   60,  220),
    "text":     (220, 220, 230),
    "dim":      (100, 105, 115),
    "blink":    (0,   220, 255),
    "green":    (40,  220, 100),
}

FONT      = cv2.FONT_HERSHEY_SIMPLEX
FONT_BOLD = cv2.FONT_HERSHEY_DUPLEX


def draw_panel(img, x, y, w, h, alpha=0.6):
    overlay = img.copy()
    cv2.rectangle(overlay, (x, y), (x+w, y+h), COLORS["panel"], -1)
    cv2.addWeighted(overlay, alpha, img, 1-alpha, 0, img)
    cv2.rectangle(img, (x, y), (x+w, y+h), COLORS["accent"], 1)


def draw_text(img, text, x, y, scale=0.55, color=None, bold=False):
    c = color or COLORS["text"]
    f = FONT_BOLD if bold else FONT
    cv2.putText(img, text, (x, y), f, scale, c, 1, cv2.LINE_AA)


def draw_bar(img, x, y, w, h, value, color, label=""):
    """Horizontal progress bar [-1, 1]."""
    mid = x + w // 2
    cv2.rectangle(img, (x, y), (x+w, y+h), COLORS["dim"], 1)
    pix = int(abs(value) * (w // 2))
    if value >= 0:
        cv2.rectangle(img, (mid, y+1), (mid+pix, y+h-1), color, -1)
    else:
        cv2.rectangle(img, (mid-pix, y+1), (mid, y+h-1), color, -1)
    cv2.line(img, (mid, y), (mid, y+h), COLORS["text"], 1)
    if label:
        draw_text(img, label, x+2, y+h-3, 0.38, COLORS["dim"])


def draw_gauge(img, cx, cy, r, value, lo, hi, label, color):
    """Semi-circular gauge."""
    angle = 180 - int(180 * (value - lo) / (hi - lo + 1e-6))
    angle = max(0, min(180, angle))
    cv2.ellipse(img, (cx, cy), (r, r), 0, 180, 360, COLORS["dim"], 2)
    cv2.ellipse(img, (cx, cy), (r, r), 0, 180, 360 - angle, color, 3)
    draw_text(img, f"{value:+.1f}", cx-18, cy+12, 0.50, color, True)
    draw_text(img, label, cx - len(label)*4, cy+28, 0.38, COLORS["dim"])


def draw_crosshair(img, cx, cy, size=12, color=(0,200,160), thickness=2):
    cv2.line(img, (cx-size, cy), (cx+size, cy), color, thickness, cv2.LINE_AA)
    cv2.line(img, (cx, cy-size), (cx, cy+size), color, thickness, cv2.LINE_AA)
    cv2.circle(img, (cx, cy), 3, color, -1, cv2.LINE_AA)


def draw_gaze_dot(img, panel_x, panel_y, panel_w, panel_h,
                  h_off, v_off, color):
    """Draw iris gaze position in a box."""
    gx = int(panel_x + panel_w/2 + h_off * panel_w * 0.4)
    gy = int(panel_y + panel_h/2 + v_off * panel_h * 0.4)
    cv2.circle(img, (gx, gy), 6, color, -1, cv2.LINE_AA)
    cv2.circle(img, (gx, gy), 8, COLORS["text"], 1, cv2.LINE_AA)


# ─────────────────────────────────────────────
#  MAIN APPLICATION
# ─────────────────────────────────────────────
def main():
    mp_face  = mp.solutions.face_mesh
    face_mesh = mp_face.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,   # enables iris landmarks
        min_detection_confidence=0.6,
        min_tracking_confidence=0.5,
    )

    cap = cv2.VideoCapture(Config.CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  Config.FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, Config.FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS,          Config.FPS_TARGET)
    cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)   # reduce latency

    img_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    img_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Camera intrinsics (approximation; good enough for most webcams)
    focal   = img_w
    cam_mat = np.array([
        [focal, 0,     img_w / 2],
        [0,     focal, img_h / 2],
        [0,     0,     1        ]
    ], dtype=np.float64)
    dist_coeffs = np.zeros((4, 1))

    # Filters
    kf_yaw   = KalmanFilter1D(Config.KALMAN_PROCESS_NOISE, Config.KALMAN_MEASURE_NOISE)
    kf_pitch = KalmanFilter1D(Config.KALMAN_PROCESS_NOISE, Config.KALMAN_MEASURE_NOISE)
    kf_roll  = KalmanFilter1D(Config.KALMAN_PROCESS_NOISE, Config.KALMAN_MEASURE_NOISE)

    kf_lh = KalmanFilter1D(1e-3, 5e-2)   # left  iris horizontal
    kf_lv = KalmanFilter1D(1e-3, 5e-2)   # left  iris vertical
    kf_rh = KalmanFilter1D(1e-3, 5e-2)   # right iris horizontal
    kf_rv = KalmanFilter1D(1e-3, 5e-2)   # right iris vertical

    ear_buf_l = MedianBuffer(Config.SMOOTH_WINDOW)
    ear_buf_r = MedianBuffer(Config.SMOOTH_WINDOW)

    blink_det = BlinkDetector()
    dwell     = DwellTracker()

    # Calibration baseline
    baseline = {"yaw": 0.0, "pitch": 0.0, "roll": 0.0}
    calibrated = False

    # State
    yaw_s = pitch_s = roll_s = 0.0
    lh_s  = lv_s  = rh_s  = rv_s  = 0.0
    ear_l = ear_r = 0.3
    prev_time = time.time()
    fps_buf   = deque(maxlen=30)

    # Log
    log_rows = []

    print("═" * 60)
    print("  Paralytic Patient Face Tracker  |  Press R to calibrate")
    print("  Q / ESC to quit  |  S to save log")
    print("═" * 60)

    # ── Create display window ──────────────────────────────────
    WIN = "Paralytic Face Tracker"
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN, 1440, 840)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        now   = time.time()
        now_ms = now * 1000.0

        # FPS
        dt = now - prev_time
        prev_time = now
        fps_buf.append(1.0 / (dt + 1e-6))
        fps = np.mean(fps_buf)

        # ── MediaPipe ─────────────────────────────────────────
        rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = face_mesh.process(rgb)

        active_cmd = "neutral"

        if result.multi_face_landmarks:
            lm = result.multi_face_landmarks[0].landmark

            # ── Head Pose ─────────────────────────────────────
            pose = get_head_pose(lm, img_w, img_h, cam_mat, dist_coeffs)
            if pose:
                yaw_raw, pitch_raw, roll_raw = pose
                yaw_raw   -= baseline["yaw"]
                pitch_raw -= baseline["pitch"]
                roll_raw  -= baseline["roll"]

                if not calibrated:
                    # Auto-calibrate on first detection
                    kf_yaw.reset(yaw_raw); kf_pitch.reset(pitch_raw); kf_roll.reset(roll_raw)
                    calibrated = True

                yaw_s   = kf_yaw.update(yaw_raw)
                pitch_s = kf_pitch.update(pitch_raw)
                roll_s  = kf_roll.update(roll_raw)

            # ── EAR ───────────────────────────────────────────
            raw_ear_l = eye_aspect_ratio(lm, LEFT_EAR_IDX,  img_w, img_h)
            raw_ear_r = eye_aspect_ratio(lm, RIGHT_EAR_IDX, img_w, img_h)
            ear_l = ear_buf_l.update(raw_ear_l)
            ear_r = ear_buf_r.update(raw_ear_r)
            ear_avg = (ear_l + ear_r) / 2.0

            bstate = blink_det.update(ear_avg, now_ms)

            # ── Iris Gaze ─────────────────────────────────────
            try:
                lh_raw, lv_raw = iris_gaze(lm, LEFT_IRIS,  LEFT_EYE,  img_w, img_h)
                rh_raw, rv_raw = iris_gaze(lm, RIGHT_IRIS, RIGHT_EYE, img_w, img_h)
                lh_s = kf_lh.update(lh_raw); lv_s = kf_lv.update(lv_raw)
                rh_s = kf_rh.update(rh_raw); rv_s = kf_rv.update(rv_raw)
            except Exception:
                pass

            # Average both eyes for gaze
            gaze_h = (lh_s + rh_s) / 2.0
            gaze_v = (lv_s + rv_s) / 2.0

            # ── Command Interpretation ────────────────────────
            if bstate.double_blink:
                active_cmd = "double_blink"
            elif bstate.is_closed:
                active_cmd = "blink"
            elif abs(yaw_s) > Config.HEAD_THRESHOLD_YAW:
                active_cmd = "head_right" if yaw_s > 0 else "head_left"
            elif abs(pitch_s) > Config.HEAD_THRESHOLD_PITCH:
                active_cmd = "head_down" if pitch_s > 0 else "head_up"
            elif abs(gaze_h) > Config.IRIS_THRESHOLD_H:
                active_cmd = "eye_right" if gaze_h > 0 else "eye_left"
            elif abs(gaze_v) > Config.IRIS_THRESHOLD_V:
                active_cmd = "eye_down" if gaze_v > 0 else "eye_up"

            confirmed = dwell.update(active_cmd, now)

            # Log
            log_rows.append({
                "time_s":    f"{now:.3f}",
                "fps":       f"{fps:.1f}",
                "yaw":       f"{yaw_s:.2f}",
                "pitch":     f"{pitch_s:.2f}",
                "roll":      f"{roll_s:.2f}",
                "gaze_h":    f"{gaze_h:.3f}",
                "gaze_v":    f"{gaze_v:.3f}",
                "ear_l":     f"{ear_l:.3f}",
                "ear_r":     f"{ear_r:.3f}",
                "blinks":    bstate.blink_count,
                "command":   active_cmd,
                "confirmed": confirmed,
            })

        # ── Build Display ─────────────────────────────────────
        display = np.zeros((840, 1440, 3), dtype=np.uint8)

        # Camera feed (left side, scaled)
        feed_h, feed_w = 540, 960
        cam_view = cv2.resize(frame, (feed_w, feed_h))
        display[40:40+feed_h, 40:40+feed_w] = cam_view

        # Camera border
        cv2.rectangle(display, (40, 40), (40+feed_w, 40+feed_h), COLORS["accent"], 2)

        # ── Right Panel ───────────────────────────────────────
        PX, PY, PW = 1040, 20, 380
        draw_panel(display, PX, PY, PW, 800)

        draw_text(display, "PATIENT TRACKER", PX+12, PY+26, 0.65, COLORS["accent"], True)
        draw_text(display, f"FPS {fps:5.1f}", PX+240, PY+26, 0.50, COLORS["dim"])

        # ── Head Pose Gauges ──────────────────────────────────
        draw_text(display, "HEAD ORIENTATION", PX+12, PY+60, 0.50, COLORS["dim"])
        draw_gauge(display, PX+65,  PY+115, 40, yaw_s,   -45, 45, "YAW",   COLORS["warning"])
        draw_gauge(display, PX+195, PY+115, 40, pitch_s, -30, 30, "PITCH", COLORS["green"])
        draw_gauge(display, PX+325, PY+115, 40, roll_s,  -30, 30, "ROLL",  COLORS["accent"])

        # ── Head direction bars ───────────────────────────────
        draw_text(display, "HEAD  H", PX+12, PY+168, 0.40, COLORS["dim"])
        draw_bar(display, PX+12, PY+172, PW-24, 12,
                 yaw_s / 45.0, COLORS["warning"], "")
        draw_text(display, "HEAD  V", PX+12, PY+198, 0.40, COLORS["dim"])
        draw_bar(display, PX+12, PY+202, PW-24, 12,
                 pitch_s / 30.0, COLORS["green"], "")

        # ── EAR Bars ──────────────────────────────────────────
        draw_text(display, "EYE ASPECT RATIO", PX+12, PY+232, 0.50, COLORS["dim"])
        for i, (label, ear_val) in enumerate([("L", ear_l), ("R", ear_r)]):
            bx = PX + 12 + i * 188
            bw = 176
            pct = min(ear_val / 0.4, 1.0)
            col = COLORS["blink"] if ear_val < Config.EAR_BLINK_THRESHOLD else COLORS["accent"]
            cv2.rectangle(display, (bx, PY+240), (bx+bw, PY+258), COLORS["dim"], 1)
            cv2.rectangle(display, (bx, PY+240), (bx+int(bw*pct), PY+258), col, -1)
            draw_text(display, f"{label}: {ear_val:.3f}", bx+4, PY+270, 0.45, col)

        # ── Blink indicator ───────────────────────────────────
        bstate_ref = blink_det.state
        blink_col  = COLORS["blink"] if bstate_ref.is_closed else COLORS["dim"]
        cv2.circle(display, (PX+30, PY+298), 12, blink_col, -1, cv2.LINE_AA)
        draw_text(display, f"Blinks: {bstate_ref.blink_count}", PX+50, PY+302, 0.50, COLORS["text"])

        # ── Gaze boxes ────────────────────────────────────────
        draw_text(display, "EYE GAZE", PX+12, PY+328, 0.50, COLORS["dim"])
        for i, (label, gh, gv) in enumerate([
            ("LEFT",  lh_s, lv_s),
            ("RIGHT", rh_s, rv_s),
        ]):
            gx = PX + 12 + i * 190
            gy = PY + 335
            gw, gh_dim = 175, 100
            draw_panel(display, gx, gy, gw, gh_dim, 0.8)
            draw_text(display, label, gx+4, gy+14, 0.40, COLORS["dim"])
            # Crosshair
            cx_p = gx + gw // 2
            cy_p = gy + gh_dim // 2
            draw_crosshair(display, cx_p, cy_p, 8, COLORS["dim"])
            # Dot
            dot_x = int(cx_p + gh * gw * 0.38)
            dot_y = int(cy_p + gv * gh_dim * 0.38)
            cv2.circle(display, (dot_x, dot_y), 6, COLORS["accent"], -1, cv2.LINE_AA)

        # ── Gaze H/V bars ─────────────────────────────────────
        avg_gh = (lh_s + rh_s) / 2.0
        avg_gv = (lv_s + rv_s) / 2.0
        draw_text(display, "GAZE  H", PX+12, PY+448, 0.40, COLORS["dim"])
        draw_bar(display, PX+12, PY+452, PW-24, 12, avg_gh, COLORS["accent"])
        draw_text(display, "GAZE  V", PX+12, PY+476, 0.40, COLORS["dim"])
        draw_bar(display, PX+12, PY+480, PW-24, 12, avg_gv, COLORS["accent"])

        # ── Command Box ───────────────────────────────────────
        draw_text(display, "ACTIVE COMMAND", PX+12, PY+512, 0.50, COLORS["dim"])
        cmd_col = COLORS["accent"] if active_cmd != "neutral" else COLORS["dim"]
        draw_panel(display, PX+12, PY+518, PW-24, 48, 0.85)
        cmd_label = COMMANDS.get(active_cmd, active_cmd)
        draw_text(display, cmd_label, PX+20, PY+548, 0.72, cmd_col, True)

        # Dwell progress bar
        prog = dwell.progress(now)
        pbar_w = int((PW - 24) * prog)
        cv2.rectangle(display, (PX+12, PY+568), (PX+12+PW-24, PY+574), COLORS["dim"], 1)
        if pbar_w > 0:
            cv2.rectangle(display, (PX+12, PY+568), (PX+12+pbar_w, PY+574), cmd_col, -1)
        draw_text(display, f"Dwell: {prog*100:.0f}%", PX+12, PY+590, 0.42, COLORS["dim"])

        if dwell.confirmed:
            draw_text(display, "✓ CONFIRMED", PX+120, PY+590, 0.55, COLORS["green"], True)

        # ── Confirmed command history (bottom left) ───────────
        draw_text(display, "CONTROLS: R=Recalibrate  S=Save Log  Q/ESC=Quit",
                  PX+12, PY+620, 0.40, COLORS["dim"])
        draw_text(display, f"YAW={yaw_s:+.1f}° PITCH={pitch_s:+.1f}° ROLL={roll_s:+.1f}°",
                  PX+12, PY+645, 0.42, COLORS["text"])

        # ── Title bar on camera ───────────────────────────────
        cv2.rectangle(display, (40, 10), (650, 38), COLORS["panel"], -1)
        draw_text(display, "LIVE CAMERA FEED  (Kalman-filtered landmarks)", 50, 30,
                  0.50, COLORS["accent"])

        # Face tracking overlay on camera view
        if result.multi_face_landmarks:
            # Draw key landmarks
            for idx in FACE_LANDMARK_IDS:
                px_ = int(result.multi_face_landmarks[0].landmark[idx].x * img_w)
                py_ = int(result.multi_face_landmarks[0].landmark[idx].y * img_h)
                # Map to display coords
                dpx = int(40 + px_ * feed_w / img_w)
                dpy = int(40 + py_ * feed_h / img_h)
                cv2.circle(display, (dpx, dpy), 3, COLORS["accent"], -1, cv2.LINE_AA)

            # Draw iris on display
            for iris_grp, eye_col in [
                (LEFT_IRIS,  COLORS["blink"]),
                (RIGHT_IRIS, COLORS["warning"]),
            ]:
                for idx in iris_grp:
                    px_ = int(result.multi_face_landmarks[0].landmark[idx].x * img_w)
                    py_ = int(result.multi_face_landmarks[0].landmark[idx].y * img_h)
                    dpx = int(40 + px_ * feed_w / img_w)
                    dpy = int(40 + py_ * feed_h / img_h)
                    cv2.circle(display, (dpx, dpy), 2, eye_col, -1, cv2.LINE_AA)

        # ── Bottom strip ──────────────────────────────────────
        y_bot = 600
        draw_panel(display, 40, y_bot, feed_w, 200, 0.7)
        draw_text(display, "DATA STREAM", 52, y_bot+20, 0.50, COLORS["dim"])
        labels = [
            ("HEAD YAW",   f"{yaw_s:+7.2f}°",  COLORS["warning"]),
            ("HEAD PITCH", f"{pitch_s:+7.2f}°", COLORS["green"]),
            ("HEAD ROLL",  f"{roll_s:+7.2f}°",  COLORS["accent"]),
            ("GAZE H",     f"{avg_gh:+6.3f}",   COLORS["accent"]),
            ("GAZE V",     f"{avg_gv:+6.3f}",   COLORS["warning"]),
            ("EAR-L",      f"{ear_l:.3f}",       COLORS["blink"]),
            ("EAR-R",      f"{ear_r:.3f}",       COLORS["blink"]),
            ("BLINKS",     str(bstate_ref.blink_count), COLORS["text"]),
        ]
        for i, (lab, val, col) in enumerate(labels):
            bx = 52 + i * 118
            draw_text(display, lab, bx, y_bot+44, 0.38, COLORS["dim"])
            draw_text(display, val, bx, y_bot+68, 0.55, col, True)

        # Active command large display
        draw_text(display, COMMANDS.get(active_cmd, ""), 52, y_bot+120, 0.80,
                  cmd_col, True)

        cv2.imshow(WIN, display)

        # ── Key Handling ──────────────────────────────────────
        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):
            break
        elif key == ord('r'):
            # Recalibrate
            if result and result.multi_face_landmarks:
                pose = get_head_pose(
                    result.multi_face_landmarks[0].landmark,
                    img_w, img_h, cam_mat, dist_coeffs
                )
                if pose:
                    baseline["yaw"], baseline["pitch"], baseline["roll"] = pose
                    kf_yaw.reset(); kf_pitch.reset(); kf_roll.reset()
                    print("  ✓ Calibration reset.")
        elif key == ord('s'):
            if log_rows:
                with open(Config.LOG_FILE, "w", newline="") as f:
                    w = csv.DictWriter(f, fieldnames=log_rows[0].keys())
                    w.writeheader(); w.writerows(log_rows)
                print(f"  ✓ Log saved → {Config.LOG_FILE}  ({len(log_rows)} rows)")

    cap.release()
    cv2.destroyAllWindows()
    face_mesh.close()
    print("Session ended.")


if __name__ == "__main__":
    main()