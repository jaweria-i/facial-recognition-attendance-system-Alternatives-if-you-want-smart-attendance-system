import cv2
import numpy as np
import os
import csv
from datetime import datetime

# Load trained model
recognizer = cv2.face.LBPHFaceRecognizer_create()
trainer_file = r"C:/Users/SYS/PycharmProjects/FRAS/TrainingImageLabel/trainer.yml"
if not os.path.exists(trainer_file):
    print("Trainer.yml file not found! Check the path.")
    exit()
recognizer.read(trainer_file)

# Load Haarcascade
face_cascade = cv2.CascadeClassifier(r"C:/Users/SYS/PycharmProjects/FRAS/haarcascade_frontalface_default.xml")
if face_cascade.empty():
    print("Haarcascade file not found!")
    exit()

# ID → Name mapping
names = {4: "Maria Izhar", 5: "Jaweria Izhar", 17: "Barira Izhar"}  # Add all IDs & names here

# Open camera
cap = cv2.VideoCapture(0)

# Attendance CSV
attendance_file = 'Attendance.csv'
if not os.path.exists(attendance_file):
    with open(attendance_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Date', 'ID', 'Name', 'Time', 'Status'])

def mark_attendance(id_, name):
    now = datetime.now()
    date_str = now.strftime('%d-%m-%Y')
    time_str = now.strftime('%H:%M:%S')

    # Make Attendance folder
    if not os.path.exists("Attendance"):
        os.makedirs("Attendance")

    attendance_file = f"Attendance/Attendance_{date_str}.csv"

    # Create file if not exists
    if not os.path.exists(attendance_file):
        with open(attendance_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Date', 'ID', 'Name', 'Time', 'Status'])

    # Check if already marked today
    already_marked = False
    rows = []

    with open(attendance_file, 'r', newline='') as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            rows.append(row)
            if row[1] == str(id_):
                already_marked = True

    if not already_marked:
        with open(attendance_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([date_str, id_, name, time_str, 'Present'])

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        id_, conf = recognizer.predict(gray[y:y + h, x:x + w])
        if conf < 80:
            name = names.get(id_, "Unknown")
            cv2.putText(frame, name, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            mark_attendance(id_, name)
        else:
            cv2.putText(frame, "Unknown", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    cv2.imshow('Face Attendance', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
