# AI Powered Workplace Monitoring System

A computer-vision based attendance and presence-monitoring system built with
OpenCV. It recognizes registered people's faces through a webcam, tracks how
long each person is present during a session, and generates an attendance
report — gated behind a simple teacher/admin login.

## Features

- **Face enrollment** — capture guided face samples (straight, left, right)
  for each person via `src/face_taker.py`, with optional hand-gesture capture
  using MediaPipe.
- **Model training** — train an OpenCV LBPH face recognizer on the captured
  dataset via `src/face_train.py`.
- **Live monitoring** — `src/main.py` provides a Tkinter login screen for a
  teacher/admin, then runs real-time face recognition over a webcam feed for
  a configurable session length, tracking per-person presence time.
- **Attendance report** — at the end of a session, each recognized person is
  marked `Present` or `Absent` based on the percentage of the session they
  were detected, and the result is saved as JSON.
- **Voice feedback** — spoken prompts and status messages via `pyttsx3`.
- **Standalone recognizer** — `src/face_recognizer.py` runs recognition with
  prediction smoothing, independent of the login flow.

## Project Structure

```
AI_Powered_Workplace_Monitoring_System/
├── src/
│   ├── main.py              # Teacher login GUI + monitoring session + report
│   ├── face_taker.py        # Guided face & hand-gesture dataset capture
│   ├── face_train.py        # Trains the LBPH face recognizer
│   └── face_recognizer.py   # Standalone recognizer with smoothing
├── assets/
│   └── haarcascade_frontalface_default.xml
├── dataset/                 # Generated face/hand images (gitignored)
├── models/                  # Generated trainer_face.yml, label_map.json (gitignored)
├── data/                    # Generated teacher.db, names.json, attendance_report.json (gitignored)
├── requirements.txt
├── LICENSE
└── README.md
```

`dataset/`, `models/`, and `data/` are created automatically at runtime and
are excluded from version control since they contain personal biometric data
and local credentials.

## Getting Started

### Prerequisites

- Python 3.9+
- A working webcam
- macOS/Linux/Windows with audio output (for voice prompts)

### Installation

```bash
git clone https://github.com/Tarun7981448226/AI_Powered_Workplace_Monitoring_System.git
cd AI_Powered_Workplace_Monitoring_System

python3 -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate

pip install -r requirements.txt
```

### Usage

All commands should be run from the project root.

**1. Enroll users (capture face & hand data)**

```bash
python src/face_taker.py
```

Follow the on-screen/voice prompts for each person: look straight, then turn
head left and right, followed by a short hand-gesture capture.

**2. Train the recognition model**

```bash
python src/face_train.py
```

This produces `models/trainer_face.yml` and `models/label_map.json`.

**3. Run the monitoring system**

```bash
python src/main.py
```

On first run you'll be asked to create a teacher/admin account (email +
password). On subsequent runs, log in and set a session duration (HH:MM:SS)
to start monitoring. Press `ESC` in the camera window to stop early. A
report is written to `data/attendance_report.json` at the end of the
session.

**4. (Optional) Run the standalone recognizer**

```bash
python src/face_recognizer.py
```

## How It Works

1. Faces are detected with a Haar Cascade classifier and normalized
   (resized + histogram-equalized) before being fed to an OpenCV LBPH
   (`cv2.face.LBPHFaceRecognizer`) model.
2. During a monitoring session, each recognized identity accumulates
   presence time based on how consistently it's detected across frames.
3. At the end of the session, a person is marked **Present** if their
   tracked presence time is at least 80% of the total session length,
   otherwise **Absent**.

## Tech Stack

- [OpenCV](https://opencv.org/) (`opencv-contrib-python`) — face detection & LBPH recognition
- [MediaPipe](https://developers.google.com/mediapipe) — hand landmark detection
- [Pillow](https://python-pillow.org/) / [NumPy](https://numpy.org/) — image processing
- [pyttsx3](https://pypi.org/project/pyttsx3/) — offline text-to-speech
- SQLite (`sqlite3`) — teacher/admin credential storage
- Tkinter — login GUI

## Notes & Limitations

- Credentials are stored in plain text in a local SQLite database — this is
  a learning/demo project, not production-ready authentication.
- Face recognition accuracy depends heavily on lighting, camera quality, and
  the number/diversity of enrolled samples per person.
- All biometric and personal data stays local; nothing is uploaded anywhere.

## License

This project is licensed under the [MIT License](LICENSE).
