import cv2
import json
import os
import time
from collections import defaultdict, deque

# -----------------------------
# Paths
# -----------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")

TRAINER_FILE = os.path.join(MODELS_DIR, "trainer_face.yml")
LABEL_MAP_FILE = os.path.join(MODELS_DIR, "label_map.json")

# -----------------------------
# Load trained recognizer
# -----------------------------
recognizer = cv2.face.LBPHFaceRecognizer_create(
    radius=1,
    neighbors=8,
    grid_x=8,
    grid_y=8
)

recognizer.read(TRAINER_FILE)

# -----------------------------
# Load label map
# -----------------------------
with open(LABEL_MAP_FILE, "r") as f:
    label_map = json.load(f)

label_map = {int(k): v for k, v in label_map.items()}

print("[INFO] Loaded users:", label_map)

# -----------------------------
# Face detector
# -----------------------------
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# -----------------------------
# Camera
# -----------------------------
cam = cv2.VideoCapture(0)

cam.set(3,640)
cam.set(4,480)

# -----------------------------
# Tracking structures
# -----------------------------
presence_time = defaultdict(float)
last_seen = {}

# prediction smoothing
prediction_history = {}

print("\n[INFO] Face recognition started")
print("[INFO] Press ESC to stop\n")

while True:

    ret, frame = cam.read()

    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(100,100)
    )

    current_time = time.time()

    for (x,y,w,h) in faces:

        face = gray[y:y+h,x:x+w]

        # same preprocessing as training
        face = cv2.resize(face,(220,220))
        face = cv2.equalizeHist(face)

        id_, confidence = recognizer.predict(face)

        if confidence < 70:
            predicted_name = label_map.get(id_,"Unknown")
        else:
            predicted_name = "Unknown"

        # smoothing key
        key = (x//50,y//50)

        if key not in prediction_history:
            prediction_history[key] = deque(maxlen=5)

        prediction_history[key].append(predicted_name)

        history = prediction_history[key]

        stable_name = max(set(history), key=history.count)

        if history.count(stable_name) < 3:
            stable_name = "Unknown"

        # -----------------------------
        # Timer logic
        # -----------------------------
        if stable_name != "Unknown":

            if stable_name not in last_seen:
                last_seen[stable_name] = current_time

            elapsed = current_time - last_seen[stable_name]

            if elapsed < 2:
                presence_time[stable_name] += elapsed

            last_seen[stable_name] = current_time

        # -----------------------------
        # Draw box
        # -----------------------------
        if stable_name != "Unknown":
            label = f"{stable_name}"
            color = (0,255,0)
        else:
            label = "Unknown"
            color = (0,0,255)

        cv2.rectangle(frame,(x,y),(x+w,y+h),color,2)

        cv2.putText(
            frame,
            label,
            (x,y-10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            color,
            2
        )

    # -----------------------------
    # Display timers
    # -----------------------------
    y = 30

    for person,t in presence_time.items():

        total = int(t)

        minutes = total // 60
        seconds = total % 60

        cv2.putText(
            frame,
            f"{person}: {minutes:02d}:{seconds:02d}",
            (10,y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255,0,0),
            2
        )

        y += 30

    cv2.imshow("AI Monitoring System", frame)

    if cv2.waitKey(1) == 27:
        break


cam.release()
cv2.destroyAllWindows()

print("\n========== SESSION REPORT ==========\n")

for person,t in presence_time.items():

    total = int(t)

    minutes = total // 60
    seconds = total % 60

    print(f"{person} -> {minutes}m {seconds}s")

print("\n[INFO] System stopped.")