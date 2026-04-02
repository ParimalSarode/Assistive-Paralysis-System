# Assistive Paralysis Communication System

The **Assistive Paralysis Communication System** is a computer-vision-based application designed to help paralyzed or non-verbal patients communicate naturally with their caregivers. By using a standard webcam, the system tracks the patient's head movements, eye gaze, and blinks, translating these subtle physical cues into fully spoken, robust English sentences.

## Features

- **Text-to-Speech (TTS) Integration:** Commands trigger an isolated background process to speak out loud, naturally stringing sentences together in a queue without freezing the user interface.
- **Robust Blink Detection:** Specifically tuned for shallower and slower blinks typical in paralytic patients, with distinct states for single blinks, double blinks, incomplete "partial" blinks, and eye-closure fatigue monitoring.
- **Kalman-Filtered Tracking:** Uses MediaPipe to locate facial landmarks and applies highly responsive 1-D Kalman filters to smooth the signals, preventing jitter and false positives.
- **Caregiver Prompting:** Selecting certain commands triggers a list of follow-up questions for the caregiver to ask. The patient can then reliably answer using simple YES/NO blinks.
- **Dwell Tracking:** To prevent accidental triggers, sustained gaze or head tilts are required (dwell time) before a command locks in, except for instantaneous triggers like blinks.

## Project Architecture

The system is highly modular:

- `face_tracker.py`: The main executable loop. Combines vision tracking, UI drawing, and state updates.
- `config.py`: Contains all detection thresholds, sentence mappings, and layout variables.
- `vision.py`: Handles MediaPipe integrations, calculating 3D head poses (yaw/pitch/roll) and detecting intra-eye iris offsets.
- `ui.py`: Manages the rich graphical overlay, bounding box handling, and color rendering for the dashboard.
- `models.py`: State-machine classes that intelligently buffer the raw vision data and execute debounce logic (`BlinkDetector`, `SentenceBuilder`, `DwellTracker`).
- `tts.py`: Audio subsystem that queues up confirmed sentences and speaks them aloud via an isolated native Windows subprocess to prevent COM thread collision.

## Installation

1. Clone this repository.
2. In your Python environment (we recommend using a `venv`), install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   *(Main libraries include `opencv-python`, `mediapipe`, `numpy`, and `pyttsx3`)*

## Usage

To start the tracker and communication dashboard, run:
```bash
python face_tracker.py
```

### Keyboard Controls

- **`R`**: Recalibrate head pose baseline (look straight at the camera and hit R)
- **`S`**: Save the current tracking session log to a CSV (`face_tracker_log.csv`)
- **`C`**: Clear the recent sentence and command history
- **`Q`** or **`ESC`**: Quit the application

## Command Dictionary

The system interprets the following subtle cues into spoken sentences:

### Blinking & Eyes
| Cue | Spoken Sentence |
| :--- | :--- |
| **Single Blink** | "Yes — I agree, that is correct." |
| **Double Blink** | "No — that is not correct, I disagree." |
| **Partial/Slow Blink** | "I am trying to respond — please wait a moment." |
| **Eyes closed > 2s** | "My eyes are closed — I may be resting or asleep." |

### Head Movement
| Cue | Spoken Sentence |
| :--- | :--- |
| **Nod Up** | "I feel better, or I need more of this." |
| **Nod Down** | "I feel worse, or I need less of this." |
| **Turn Left** | "Go back — show me the previous option." |
| **Turn Right** | "Continue — show me the next option." |
| **Tilt Left** | "I am not sure — maybe, or possibly." |
| **Tilt Right** | "Please repeat that — I did not understand." |

### Eye Gaze
| Cue | Spoken Sentence |
| :--- | :--- |
| **Look Up** | "I need something — please ask me what." |
| **Look Down** | "I am tired or feeling sleepy right now." |
| **Look Left** | "I have pain or discomfort on my left side." |
| **Look Right** | "I have pain or discomfort on my right side." |
