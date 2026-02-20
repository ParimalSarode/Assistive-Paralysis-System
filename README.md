
# Face Tracker

A Python application for real-time face detection and tracking.

## Overview

`face_tracker.py` is the main module that provides face tracking functionality using computer vision techniques.

## Features
- Multiple face detection
- Customizable detection parameters
- Support for video and camera input
- Efficient frame processing
- Real-time face detection
- Face tracking across frames
- Easy-to-use API

## Requirements

- Python 3.7+
- OpenCV
- NumPy
- Mediapipe

## Installation

```bash
pip install -r requirements.txt
```
OR
```bash
uv add -r requirements.txt
```

## Usage

```python
from face_tracker import FaceTracker

tracker = FaceTracker()
tracker.run()
```

## Project Structure

```
.
├── face_tracker.py    # Main module
└── README.md          # This file
```

## License

MIT
