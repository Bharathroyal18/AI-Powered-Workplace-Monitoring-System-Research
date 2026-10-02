import cv2
import os
import json
import time
import pyttsx3
import mediapipe as mp


# -----------------------------
# Paths
# -----------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")


# -----------------------------
# Text to Speech Setup
# -----------------------------
engine = pyttsx3.init()
engine.setProperty('rate',150)

def speak(text):
    engine.say(text)
    engine.runAndWait()


# -----------------------------
# Create folders
# -----------------------------
def create_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)


# -----------------------------
# Save user info
# -----------------------------
def save_user_info(user_id,name,hand_movement,json_file):

    data={}

    if os.path.exists(json_file):
        try:
            with open(json_file,"r") as f:
                data=json.load(f)
        except:
            data={}

    data[str(user_id)] = {
        "name":name,
        "hand_movement":hand_movement
    }

    with open(json_file,"w") as f:
        json.dump(data,f,indent=4)


# -----------------------------
# FACE CAPTURE
# -----------------------------
def capture_faces(directory):

    face_cascade=cv2.CascadeClassifier(
        cv2.data.haarcascades+"haarcascade_frontalface_default.xml"
    )

    cam=cv2.VideoCapture(0)

    cam.set(cv2.CAP_PROP_FRAME_WIDTH,1280)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT,720)

    # warmup camera
    time.sleep(2)

    phases=[
        ("Look straight",20),
        ("Turn head left slowly",5),
        ("Turn head right slowly",5)
    ]

    total_count=0

    print("\n[INFO] Guided capture starting...")

    for instruction,limit in phases:

        speak(instruction)

        print("[INFO]",instruction)

        # time to move head
        time.sleep(3)

        count=0

        while count < limit:

            ret,frame=cam.read()

            if not ret:
                break

            gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

            faces=face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.2,
                minNeighbors=6,
                minSize=(120,120)
            )

            for (x,y,w,h) in faces:

                face=gray[y:y+h,x:x+w]

                # normalize face
                face=cv2.resize(face,(220,220))
                face=cv2.equalizeHist(face)

                total_count+=1
                count+=1

                file_path=os.path.join(directory,f"{total_count}.jpg")

                cv2.imwrite(file_path,face)

                cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)

                break

            cv2.putText(
                frame,
                f"{instruction} ({count}/{limit})",
                (20,40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0,255,255),
                2
            )

            cv2.imshow("Face Capture",frame)

            # slower capture to avoid duplicates
            if cv2.waitKey(400)==27:
                break

    cam.release()
    cv2.destroyAllWindows()

    speak("Face capture complete")

    print("[INFO] Captured",total_count,"images.")


# -----------------------------
# HAND CAPTURE
# -----------------------------
def capture_hand_movements(directory):

    mp_hands=mp.solutions.hands

    hands=mp_hands.Hands(min_detection_confidence=0.6)

    mp_draw=mp.solutions.drawing_utils

    cam=cv2.VideoCapture(0)

    print("\n[INFO] Capturing hand gestures...")

    count=0

    while count < 30:

        ret,frame=cam.read()

        if not ret:
            break

        img_rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)

        results=hands.process(img_rgb)

        if results.multi_hand_landmarks:

            for hand_landmarks in results.multi_hand_landmarks:

                mp_draw.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS
                )

                count+=1

                file_path=os.path.join(directory,f"{count}.jpg")

                cv2.imwrite(file_path,frame)

        cv2.imshow("Hand Capture",frame)

        if cv2.waitKey(1)==27:
            break

    cam.release()
    cv2.destroyAllWindows()

    print("[INFO] Hand gesture capture complete.")


# -----------------------------
# MAIN PROGRAM
# -----------------------------
if __name__=="__main__":

    dataset_dir = os.path.join(BASE_DIR, "dataset")
    face_main_dir = os.path.join(dataset_dir, "faces")
    hand_main_dir = os.path.join(dataset_dir, "hands")

    user_info_file = os.path.join(DATA_DIR, "names.json")

    create_directory(DATA_DIR)
    create_directory(dataset_dir)
    create_directory(face_main_dir)
    create_directory(hand_main_dir)

    user_id=0

    while True:

        print("\n========== NEW USER ==========")

        name=input("Enter user name: ").strip()

        if not name:
            print("Name cannot be empty")
            continue

        hand_move=input("Enter hand gesture name: ").strip()

        if not hand_move:
            print("Hand gesture cannot be empty")
            continue

        face_dir=os.path.join(face_main_dir,name)
        hand_dir=os.path.join(hand_main_dir,name)

        create_directory(face_dir)
        create_directory(hand_dir)

        save_user_info(user_id,name,hand_move,user_info_file)

        capture_faces(face_dir)
        capture_hand_movements(hand_dir)

        print("\n[INFO] Data saved successfully!")

        user_id+=1

        again=input("\nAdd another user? (y/n): ").lower()

        if again!="y":
            print("\n[INFO] Data collection finished.")
            break