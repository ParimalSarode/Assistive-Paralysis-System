import numpy as np
import cv2

FACE_3D_MODEL = np.array([
    [0.0,    0.0,   0.0  ],
    [0.0, -330.0, -65.0  ],
    [-225.0, 170.0,-135.0],
    [ 225.0, 170.0,-135.0],
    [-150.0,-150.0,-125.0],
    [ 150.0,-150.0,-125.0],
], dtype=np.float64)

FACE_LANDMARK_IDS = [1,152,263,33,287,57]

LEFT_IRIS     = [474,475,476,477]
RIGHT_IRIS    = [469,470,471,472]
LEFT_EYE      = [362,382,381,380,374,373,390,249,263,466,388,387,386,385,384,398]
RIGHT_EYE     = [33,  7,163,144,145,153,154,155,133,173,157,158,159,160,161,246]
LEFT_EAR_IDX  = [362,385,387,263,373,380]
RIGHT_EAR_IDX = [33, 160,158,133,153,144]

def eye_aspect_ratio(landmarks, indices, img_w, img_h):
    pts = np.array([[landmarks[i].x*img_w, landmarks[i].y*img_h] for i in indices], dtype=float)
    A = np.linalg.norm(pts[1]-pts[5])
    B = np.linalg.norm(pts[2]-pts[4])
    C = np.linalg.norm(pts[0]-pts[3])
    return (A+B)/(2.0*C+1e-6)

def get_head_pose(landmarks, img_w, img_h, cam_mat, dist_c):
    pts = np.array([[landmarks[i].x*img_w, landmarks[i].y*img_h] for i in FACE_LANDMARK_IDS], dtype=np.float64)
    ok, rvec, tvec = cv2.solvePnP(FACE_3D_MODEL, pts, cam_mat, dist_c, flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok: return None
    rmat,_ = cv2.Rodrigues(rvec)
    _,_,_,_,_,_,euler = cv2.decomposeProjectionMatrix(np.hstack([rmat,tvec]))
    return float(euler[1]), float(euler[0]), float(euler[2])

def iris_gaze(landmarks, iris_ids, eye_ids, img_w, img_h):
    ip = np.array([[landmarks[i].x*img_w,landmarks[i].y*img_h] for i in iris_ids])
    ep = np.array([[landmarks[i].x*img_w,landmarks[i].y*img_h] for i in eye_ids])
    icx,icy = ip[:,0].mean(),ip[:,1].mean()
    ecx=(ep[:,0].min()+ep[:,0].max())/2; ecy=(ep[:,1].min()+ep[:,1].max())/2
    ehw=(ep[:,0].max()-ep[:,0].min())/2+1e-6
    ehh=(ep[:,1].max()-ep[:,1].min())/2+1e-6
    return float(np.clip((icx-ecx)/ehw,-1,1)), float(np.clip((icy-ecy)/ehh,-1,1))
