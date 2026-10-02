import cv2
import json
import os
import sqlite3
import time
import tkinter as tk
from tkinter import messagebox
from collections import defaultdict
import pyttsx3


# -------------------------
# Paths
# -------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

TEACHER_DB = os.path.join(DATA_DIR, "teacher.db")
TRAINER_FILE = os.path.join(MODELS_DIR, "trainer_face.yml")
LABEL_MAP_FILE = os.path.join(MODELS_DIR, "label_map.json")
ATTENDANCE_REPORT_FILE = os.path.join(DATA_DIR, "attendance_report.json")


# -------------------------
# Voice system
# -------------------------
engine = pyttsx3.init()
engine.setProperty("rate",150)

def speak(text):
    engine.say(text)
    engine.runAndWait()


# -------------------------
# Database setup
# -------------------------
conn = sqlite3.connect(TEACHER_DB)
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS teacher(
email TEXT,
password TEXT
)
""")

conn.commit()

cursor.execute("SELECT * FROM teacher")
teacher = cursor.fetchone()


# -------------------------
# Face monitoring
# -------------------------
def start_monitoring(total_seconds):

    required_time = total_seconds * 0.8

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(TRAINER_FILE)

    with open(LABEL_MAP_FILE,"r") as f:
        label_map = json.load(f)

    label_map = {int(k):v for k,v in label_map.items()}

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades+"haarcascade_frontalface_default.xml"
    )

    cam = cv2.VideoCapture(0)

    presence_time = defaultdict(float)
    last_seen = {}

    start_time = time.time()

    print("\nMonitoring started\n")

    while True:

        ret,frame = cam.read()

        if not ret:
            break

        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(
            gray,
            1.2,
            5,
            minSize=(100,100)
        )

        current_time = time.time()

        for (x,y,w,h) in faces:

            face = gray[y:y+h,x:x+w]

            face = cv2.resize(face,(220,220))
            face = cv2.equalizeHist(face)

            id_,confidence = recognizer.predict(face)

            if confidence < 70:
                name = label_map.get(id_,"Unknown")
            else:
                name = "Unknown"

            if name != "Unknown":

                if name not in last_seen:
                    last_seen[name] = current_time

                elapsed = current_time - last_seen[name]

                if elapsed < 2:
                    presence_time[name] += elapsed

                last_seen[name] = current_time

            color = (0,255,0) if name!="Unknown" else (0,0,255)

            cv2.rectangle(frame,(x,y),(x+w,y+h),color,2)

            cv2.putText(
                frame,
                name,
                (x,y-10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                color,
                2
            )

        y = 30

        for person,t in presence_time.items():

            sec = int(t)

            cv2.putText(
                frame,
                f"{person}: {sec}s",
                (10,y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255,0,0),
                2
            )

            y += 30

        cv2.imshow("AI Monitoring System",frame)

        if cv2.waitKey(1) == 27:
            break

        if current_time - start_time >= total_seconds:
            break


    cam.release()
    cv2.destroyAllWindows()

    print("\n===== FINAL REPORT =====\n")

    report = {}

    for person,time_spent in presence_time.items():

        if time_spent >= required_time:
            status = "Present"
        else:
            status = "Absent"

        report[person] = status

        print(f"{person} -> {int(time_spent)}s -> {status}")

    with open(ATTENDANCE_REPORT_FILE,"w") as f:
        json.dump(report,f,indent=4)

    speak("Monitoring finished report generated")


# -------------------------
# Login
# -------------------------
def login():

    email = email_entry.get()
    password = pass_entry.get()

    cursor.execute(
        "SELECT * FROM teacher WHERE email=? AND password=?",
        (email,password)
    )

    result = cursor.fetchone()

    if result:

        speak("Permission granted")

        messagebox.showinfo("Access","Permission Granted")

        hours = int(hour_spin.get())
        minutes = int(min_spin.get())
        seconds = int(sec_spin.get())

        total_seconds = hours*3600 + minutes*60 + seconds

        if total_seconds == 0:
            messagebox.showerror("Error","Set monitoring time")
            return

        root.destroy()

        start_monitoring(total_seconds)

    else:

        speak("Wrong password try again")

        messagebox.showerror("Error","Wrong Credentials")


# -------------------------
# Create account
# -------------------------
def create_account():

    email = email_entry.get()
    password = pass_entry.get()

    cursor.execute("INSERT INTO teacher VALUES(?,?)",(email,password))
    conn.commit()

    messagebox.showinfo("Success","Account created")

    root.destroy()

    login_window()


# -------------------------
# Login window
# -------------------------
def login_window():

    global root,email_entry,pass_entry,hour_spin,min_spin,sec_spin

    root = tk.Tk()
    root.title("AI Powered Workplace Monitoring System")
    root.geometry("350x300")

    tk.Label(root,text="Teacher Login",font=("Arial",16)).pack(pady=10)

    tk.Label(root,text="Email").pack()
    email_entry = tk.Entry(root,width=30)
    email_entry.pack()

    tk.Label(root,text="Password").pack()
    pass_entry = tk.Entry(root,show="*",width=30)
    pass_entry.pack()

    tk.Label(root,text="Session Time (HH : MM : SS)").pack(pady=10)

    time_frame = tk.Frame(root)
    time_frame.pack()

    hour_spin = tk.Spinbox(time_frame,from_=0,to=23,width=5)
    hour_spin.pack(side="left")

    tk.Label(time_frame,text=":").pack(side="left")

    min_spin = tk.Spinbox(time_frame,from_=0,to=59,width=5)
    min_spin.pack(side="left")

    tk.Label(time_frame,text=":").pack(side="left")

    sec_spin = tk.Spinbox(time_frame,from_=0,to=59,width=5)
    sec_spin.pack(side="left")

    tk.Button(root,text="Login",command=login).pack(pady=20)

    root.mainloop()


# -------------------------
# First run
# -------------------------
if teacher is None:

    root = tk.Tk()
    root.title("Create Teacher Account")
    root.geometry("350x200")

    tk.Label(root,text="Create Teacher Account",font=("Arial",14)).pack(pady=10)

    tk.Label(root,text="Email").pack()
    email_entry = tk.Entry(root,width=30)
    email_entry.pack()

    tk.Label(root,text="Password").pack()
    pass_entry = tk.Entry(root,width=30)
    pass_entry.pack()

    tk.Button(root,text="Create Account",command=create_account).pack(pady=10)

    root.mainloop()

else:

    login_window()