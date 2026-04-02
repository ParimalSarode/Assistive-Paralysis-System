import numpy as np
from collections import deque
from dataclasses import dataclass, field
from typing import List, Tuple
from config import Config, CUE_SENTENCES, CAREGIVER_QUESTIONS

class KalmanFilter1D:
    def __init__(self, process_noise=1e-3, measure_noise=1e-1):
        self.Q = process_noise; self.R = measure_noise
        self.x = np.zeros((2,1)); self.P = np.eye(2)
        self.F = np.array([[1,1],[0,1]],dtype=float)
        self.H = np.array([[1,0]],dtype=float)

    def update(self, z):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + np.eye(2)*self.Q
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T / S[0,0]
        self.x += K*(z-(self.H@self.x)[0,0])
        self.P = (np.eye(2)-K@self.H) @ self.P
        return float(self.x[0,0])

    def reset(self, value=0.0):
        self.x = np.array([[value],[0.0]]); self.P = np.eye(2)

class MedianBuffer:
    def __init__(self, size=5):
        self.buf = deque(maxlen=size)
    def update(self, v):
        self.buf.append(v); return float(np.median(self.buf))

@dataclass
class BlinkState:
    is_closed:           bool  = False
    is_partial:          bool  = False
    is_fatigue:          bool  = False
    consec_closed:       int   = 0
    closed_since_ms:     float = 0.0
    fatigue_duration_s:  float = 0.0
    blink_count:         int   = 0
    last_blink_ms:       float = 0.0
    double_blink:        bool  = False
    just_blinked:        bool  = False

class BlinkDetector:
    def __init__(self):
        self.state = BlinkState()
        self.pending_single_blink = False
        self.single_blink_timer = 0.0

    def update(self, ear: float, now_ms: float) -> BlinkState:
        s = self.state
        s.double_blink = False
        s.just_blinked = False

        if self.pending_single_blink and (now_ms - self.single_blink_timer) > Config.BLINK_DOUBLE_MS:
            s.just_blinked = True
            self.pending_single_blink = False

        if ear < Config.EAR_BLINK_THRESHOLD:
            s.is_partial    = False
            s.consec_closed += 1
            if s.consec_closed >= Config.BLINK_CONSEC_FRAMES and not s.is_closed:
                s.is_closed       = True
                s.closed_since_ms = now_ms
                s.is_fatigue      = False
            if s.is_closed:
                s.fatigue_duration_s = (now_ms - s.closed_since_ms) / 1000.0
                if s.fatigue_duration_s >= Config.BLINK_MAX_DURATION_MS / 1000.0:
                    s.is_fatigue = True
                    self.pending_single_blink = False
        elif ear < Config.EAR_PARTIAL_HIGH:
            s.is_partial = True
        elif ear < Config.EAR_OPEN_THRESHOLD:
            pass
        else:
            s.is_partial = False
            if s.is_closed:
                if not s.is_fatigue:
                    s.blink_count  += 1
                    dt = now_ms - s.last_blink_ms
                    if 0 < dt <= Config.BLINK_DOUBLE_MS:
                        s.double_blink = True
                        s.just_blinked = False
                        self.pending_single_blink = False
                    else:
                        self.pending_single_blink = True
                        self.single_blink_timer = now_ms
                    s.last_blink_ms = now_ms
            s.is_closed          = False
            s.is_fatigue         = False
            s.consec_closed      = 0
            s.closed_since_ms    = 0.0
            s.fatigue_duration_s = 0.0
        return s

@dataclass
class DwellTracker:
    current_cmd: str   = "neutral"
    start_time:  float = 0.0
    confirmed:   bool  = False

    def update(self, cmd, now):
        if cmd != self.current_cmd:
            self.current_cmd = cmd; self.start_time = now; self.confirmed = False
        elif not self.confirmed and (now-self.start_time) >= Config.DWELL_TIME_S:
            self.confirmed = True; return True
        return False

    def progress(self, now):
        return 1.0 if self.confirmed else min(1.0,(now-self.start_time)/Config.DWELL_TIME_S)

@dataclass
class SentenceBuilder:
    history:         List[Tuple[float, str, str]] = field(default_factory=list)
    last_cue:        str   = ""
    last_confirmed:  float = 0.0
    question_idx:    int   = 0
    active_category: str   = ""

    def confirm(self, cue: str, now: float):
        if cue == self.last_cue and (now - self.last_confirmed) < 1.0:
            return None
        self.last_cue      = cue
        self.last_confirmed = now

        short, sentence = CUE_SENTENCES.get(cue, ("?", "Unknown cue."))
        self.history.append((now, cue, sentence))
        if len(self.history) > 20: self.history.pop(0)

        if cue in CAREGIVER_QUESTIONS:
            self.active_category = cue
            self.question_idx    = 0
        elif cue == "blink" and self.active_category:
            self.question_idx = (self.question_idx + 1) % len(CAREGIVER_QUESTIONS[self.active_category])
        return sentence

    def current_question(self) -> str:
        if self.active_category and self.active_category in CAREGIVER_QUESTIONS:
            qs = CAREGIVER_QUESTIONS[self.active_category]
            return qs[self.question_idx % len(qs)]
        return ""

    def clear(self):
        self.history.clear()
        self.last_cue        = ""
        self.active_category = ""
        self.question_idx    = 0
