import cv2
import mediapipe as mp
import numpy as np
import math
from collections import deque
from camera import camera

# ---------------- INIT ---------------- #

mp_face_mesh = mp.solutions.face_mesh

face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)

cap = cv2.VideoCapture(camera)

baseline_yaw = None
baseline_pitch = None
baseline_eye_h = None

yaw_buffer = deque(maxlen=5)
pitch_buffer = deque(maxlen=5)

# Thresholds
YAW_THRESHOLD = 5
PITCH_THRESHOLD = 5
EYE_THRESHOLD = 0.12
BLINK_THRESHOLD = 3

print("Press 'c' to calibrate neutral.")
print("Press 'q' to quit.")

# ---------------- HEAD POSE FUNCTION ---------------- #

def get_head_pose(image_points, frame):
    h, w = frame.shape[:2]
    focal_length = w
    center = (w / 2, h / 2)

    camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1]
    ], dtype="double")

    dist_coeffs = np.zeros((4, 1))

    model_points = np.array([
        (0.0, 0.0, 0.0),
        (0.0, -63.6, -12.5),
        (-43.3, 32.7, -26.0),
        (43.3, 32.7, -26.0),
        (-28.9, -28.9, -24.1),
        (28.9, -28.9, -24.1)
    ], dtype=np.float64)

    success, rotation_vector, _ = cv2.solvePnP(
        model_points,
        image_points,
        camera_matrix,
        dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE
    )

    if not success:
        return None, None

    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)

    sy = math.sqrt(rotation_matrix[0, 0]**2 + rotation_matrix[1, 0]**2)

    x = math.atan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
    y = math.atan2(-rotation_matrix[2, 0], sy)

    pitch = np.degrees(x)
    yaw = np.degrees(y)

    return yaw, pitch

# ---------------- MAIN LOOP ---------------- #

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    direction = "NO FACE"

    if results.multi_face_landmarks:
        face = results.multi_face_landmarks[0]

        head_direction = None
        eye_direction = None

        # ---------- HEAD DETECTION ---------- #

        ids = [1, 152, 33, 263, 61, 291]
        image_points = []

        for i in ids:
            lm = face.landmark[i]
            image_points.append((int(lm.x * w), int(lm.y * h)))

        image_points = np.array(image_points, dtype="double")

        yaw, pitch = get_head_pose(image_points, frame)

        if yaw is not None and baseline_yaw is not None:
            yaw_buffer.append(yaw)
            pitch_buffer.append(pitch)

            smooth_yaw = np.mean(yaw_buffer)
            smooth_pitch = np.mean(pitch_buffer)

            delta_yaw = smooth_yaw - baseline_yaw
            delta_pitch = smooth_pitch - baseline_pitch

            if delta_yaw > YAW_THRESHOLD:
                head_direction = "HEAD RIGHT"
            elif delta_yaw < -YAW_THRESHOLD:
                head_direction = "HEAD LEFT"
            elif delta_pitch > PITCH_THRESHOLD:
                head_direction = "HEAD UP"
            elif delta_pitch < -PITCH_THRESHOLD:
                head_direction = "HEAD DOWN"

        # ---------- EYE HORIZONTAL ONLY ---------- #

        left_corner = face.landmark[33]
        right_corner = face.landmark[133]
        iris = face.landmark[468]

        lx = int(left_corner.x * w)
        rx = int(right_corner.x * w)
        ix = int(iris.x * w)

        eye_width = rx - lx

        if baseline_eye_h is not None and abs(eye_width) > 5:
            h_ratio = (ix - lx) / eye_width
            delta_h = h_ratio - baseline_eye_h

            if delta_h > EYE_THRESHOLD:
                eye_direction = "EYE RIGHT"
            elif delta_h < -EYE_THRESHOLD:
                eye_direction = "EYE LEFT"

        # ---------- BLINK ---------- #

        top = face.landmark[159]
        bottom = face.landmark[145]

        ty = int(top.y * h)
        by = int(bottom.y * h)

        if abs(by - ty) < BLINK_THRESHOLD:
            direction = "BLINK"
        else:
            # ---------- PRIORITY ---------- #
            if head_direction is not None:
                direction = head_direction
            elif eye_direction is not None:
                direction = eye_direction
            else:
                direction = "CENTER"

    cv2.putText(frame, direction, (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1, (0, 255, 0), 2)

    cv2.imshow("Assistive Stable System", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('c') and yaw is not None:
        baseline_yaw = yaw
        baseline_pitch = pitch

        if abs(eye_width) > 5:
            baseline_eye_h = (ix - lx) / eye_width

        print("Calibrated")

    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
