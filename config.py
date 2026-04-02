class Config:
    CAMERA_INDEX         = 0
    FRAME_WIDTH          = 1280
    FRAME_HEIGHT         = 720
    FPS_TARGET           = 30

    KALMAN_PROCESS_NOISE = 1e-3
    KALMAN_MEASURE_NOISE = 1e-1

    HEAD_THRESHOLD_YAW   = 15.0    
    HEAD_THRESHOLD_PITCH = 10.0
    HEAD_THRESHOLD_ROLL  = 18.0   

    EAR_BLINK_THRESHOLD  = 0.15   
    EAR_PARTIAL_HIGH     = 0.19   
    EAR_OPEN_THRESHOLD   = 0.22   

    BLINK_CONSEC_FRAMES  = 2      
    BLINK_DOUBLE_MS      = 500    
    BLINK_MAX_DURATION_MS= 2000   

    IRIS_THRESHOLD_H     = 0.15
    IRIS_THRESHOLD_V     = 0.12

    SMOOTH_WINDOW        = 5
    DWELL_TIME_S         = 1.0    

    LOG_FILE             = "face_tracker_log.csv"

CUE_SENTENCES = {
    "blink":         ("YES", "Yes — I agree, that is correct."),
    "double_blink":  ("NO", "No — that is not correct, I disagree."),
    "partial_blink": ("TRYING TO BLINK", "I am trying to respond — please wait a moment."),
    "head_left":     ("PREVIOUS / GO BACK", "Go back — show me the previous option."),
    "head_right":    ("NEXT / CONTINUE", "Continue — show me the next option."),
    "head_up":       ("BETTER / MORE", "I feel better, or I need more of this."),
    "head_down":     ("WORSE / LESS", "I feel worse, or I need less of this."),
    "tilt_left":     ("UNSURE / MAYBE", "I am not sure — maybe, or possibly."),
    "tilt_right":    ("REPEAT PLEASE", "Please repeat that — I did not understand."),
    "eye_left":      ("PAIN / DISCOMFORT LEFT", "I have pain or discomfort on my left side."),
    "eye_right":     ("PAIN / DISCOMFORT RIGHT", "I have pain or discomfort on my right side."),
    "eye_up":        ("I NEED SOMETHING", "I need something — please ask me what."),
    "eye_down":      ("I AM TIRED / SLEEPY", "I am tired or feeling sleepy right now."),
    "fatigue":       ("RESTING / ASLEEP", "My eyes are closed — I may be resting or asleep."),
    "neutral":       ("WAITING", "No signal — patient is at rest."),
}

CAREGIVER_QUESTIONS = {
    "head_right": [
        "Are you in pain?", "Do you need water?", "Do you need the bathroom?",
        "Are you too hot?", "Are you too cold?", "Do you want the TV or radio on?",
        "Do you want me to call someone?", "Do you want to rest now?",
    ],
    "head_left": [
        "Should I call the nurse?", "Do you want to change position?", "Is the light bothering you?",
        "Is there noise bothering you?", "Do you need your medication?", "Do you want food?",
        "Do you need your phone or tablet?", "Is something hurting right now?",
    ],
}

EMOTION_MAP = {
    "blink":        "Agreement / Confirmation",
    "double_blink": "Disagreement / Negation",
    "head_up":      "Positive / Improving",
    "head_down":    "Negative / Worsening",
    "eye_down":     "Fatigue / Discomfort",
    "eye_up":       "Need / Request",
    "tilt_left":    "Uncertainty",
    "tilt_right":   "Confusion / Needs repeat",
    "eye_left":     "Pain — left side",
    "eye_right":    "Pain — right side",
    "fatigue":      "Resting / Sleeping",
    "neutral":      "At rest / No signal",
}
