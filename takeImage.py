import csv
import os, cv2
import numpy as np
import pandas as pd
import datetime
import time


# take Image of user
def TakeImage(l1, l2, haarcasecade_path, trainimage_path, message, err_screen, text_to_speech):
    if (l1 == "") and (l2 == ""):
        text_to_speech("Please Enter your Enrollment Number and Name.")
    elif l1 == "":
        text_to_speech("Please Enter your Enrollment Number.")
    elif l2 == "":
        text_to_speech("Please Enter your Name.")
    else:
        try:
            cam = cv2.VideoCapture(0)
            detector = cv2.CascadeClassifier(haarcasecade_path)
            Enrollment = l1
            Name = l2
            sampleNum = 0

            # Create directory
            directory = f"{Enrollment}_{Name}"
            path = os.path.join(trainimage_path, directory)
            os.mkdir(path)

            while True:
                ret, img = cam.read()
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                faces = detector.detectMultiScale(gray, 1.3, 5)

                for (x, y, w, h) in faces:
                    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

                    sampleNum += 1
                    filename = f"{Name}_{Enrollment}_{sampleNum}.jpg"
                    filepath = os.path.join(path, filename)

                    cv2.imwrite(filepath, gray[y:y+h, x:x+w])
                    cv2.imshow("Frame", img)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
                elif sampleNum > 50:
                    break

            cam.release()
            cv2.destroyAllWindows()

            # Save student details
            row = [Enrollment, Name]
            with open("StudentDetails/studentdetails.csv", "a+", newline='') as csvFile:
                writer = csv.writer(csvFile)
                writer.writerow(row)

            res = f"Images Saved for ER No: {Enrollment} Name: {Name}"
            message.configure(text=res)
            text_to_speech(res)

        except FileExistsError:
            text_to_speech("Student Data already exists")
