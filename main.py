import cv2
import mediapipe as mp
import numpy as np
import math

# ---------------- INIT ---------------- #

mp_face_mesh = mp.solutions.face_mesh

face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

cap = cv2.VideoCapture(1)

baseline_yaw = None
baseline_pitch = None

# 3D face model points (approximate)
model_points = np.array([
    (0.0, 0.0, 0.0),        # Nose tip
    (0.0, -63.6, -12.5),    # Chin
    (-43.3, 32.7, -26.0),   # Left eye corner
    (43.3, 32.7, -26.0),    # Right eye corner
    (-28.9, -28.9, -24.1),  # Left mouth
    (28.9, -28.9, -24.1)    # Right mouth
], dtype=np.float64)

# Landmark indices for 2D mapping
LANDMARK_IDS = [1, 152, 33, 263, 61, 291]

# Eye landmarks for EAR
LEFT_EYE = [33, 160, 158, 133, 153, 144]

# ---------------- FUNCTIONS ---------------- #

def calculate_EAR(eye_points):
    A = np.linalg.norm(eye_points[1] - eye_points[5])
    B = np.linalg.norm(eye_points[2] - eye_points[4])
    C = np.linalg.norm(eye_points[0] - eye_points[3])
    ear = (A + B) / (2.0 * C)
    return ear

def get_head_pose(image_points, frame):
    h, w = frame.shape[:2]
    focal_length = w
    center = (w / 2, h / 2)

    camera_matrix = np.array(
        [[focal_length, 0, center[0]],
         [0, focal_length, center[1]],
         [0, 0, 1]], dtype="double"
    )

    dist_coeffs = np.zeros((4, 1))

    success, rotation_vector, translation_vector = cv2.solvePnP(
        model_points,
        image_points,
        camera_matrix,
        dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE
    )

    if not success:
        return None, None

    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
    sy = math.sqrt(rotation_matrix[0, 0] ** 2 + rotation_matrix[1, 0] ** 2)

    singular = sy < 1e-6

    if not singular:
        x = math.atan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
        y = math.atan2(-rotation_matrix[2, 0], sy)
        z = math.atan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
    else:
        x = math.atan2(-rotation_matrix[1, 2], rotation_matrix[1, 1])
        y = math.atan2(-rotation_matrix[2, 0], sy)
        z = 0

    pitch = np.degrees(x)
    yaw = np.degrees(y)

    return yaw, pitch

# ---------------- MAIN LOOP ---------------- #

print("Press 'c' to calibrate neutral position.")
print("Press 'q' to quit.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(frame_rgb)

    direction = "NO FACE"
    yaw = None
    pitch = None

    if results.multi_face_landmarks:
        face_landmarks = results.multi_face_landmarks[0]
        h, w = frame.shape[:2]

        landmarks_2d = []
        for id in LANDMARK_IDS:
            lm = face_landmarks.landmark[id]
            x = int(lm.x * w)
            y = int(lm.y * h)
            landmarks_2d.append((x, y))

        image_points = np.array(landmarks_2d, dtype="double")

        yaw, pitch = get_head_pose(image_points, frame)

        if yaw is not None:

            if baseline_yaw is not None:
                delta_yaw = yaw - baseline_yaw
                delta_pitch = pitch - baseline_pitch

                if delta_yaw > 5:
                    direction = "RIGHT"
                elif delta_yaw < -5:
                    direction = "LEFT"
                elif delta_pitch > 5:
                    direction = "UP"
                elif delta_pitch < -5:
                    direction = "DOWN"
                else:
                    direction = "CENTER"

            else:
                direction = "NOT CALIBRATED"

        # Blink detection
        eye_points = []
        for id in LEFT_EYE:
            lm = face_landmarks.landmark[id]
            x = int(lm.x * w)
            y = int(lm.y * h)
            eye_points.append((x, y))

        eye_points = np.array(eye_points)
        ear = calculate_EAR(eye_points)

        if ear < 0.20:
            direction = "BLINK"

    cv2.putText(frame, direction, (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1, (0, 255, 0), 2)

    cv2.imshow("Assistive Paralysis System", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('c'):
        if yaw is not None:
            baseline_yaw = yaw
            baseline_pitch = pitch
            print("Calibrated")

    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
