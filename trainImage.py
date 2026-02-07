import cv2
import os
import numpy as np
from PIL import Image

# Folder names
training_img_path = "TrainingImage"
trainer_path = "TrainingImageLabel"

# Create training folder if not exists
if not os.path.exists(training_img_path):
    os.makedirs(training_img_path)

if not os.path.exists(trainer_path):
    os.makedirs(trainer_path)

# Function to take images
def take_images():
    ID = input("Enter Enrollment No: ")
    name = input("Enter Name: ")

    cam = cv2.VideoCapture(0)
    detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

    sampleNum = 0
    folder = training_img_path + "/" + ID + "_" + name
    if not os.path.exists(folder):
        os.makedirs(folder)

    print("Camera started... Press Q to stop")

    while True:
        ret, img = cam.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = detector.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            sampleNum += 1
            cv2.imwrite(folder + "/" + name + "_" + ID + "_" + str(sampleNum) + ".jpg",
                        gray[y:y + h, x:x + w])
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

        cv2.imshow("Taking Images", img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        if sampleNum >= 50:
            break

    cam.release()
    cv2.destroyAllWindows()
    print("Images Saved Successfully!")


# Function to train images
def train_images():
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    CASCADE_PATH = r"C:/Users/SYS/PycharmProjects/FRAS/haarcascade_frontalface_default.xml"
    detector = cv2.CascadeClassifier(CASCADE_PATH)

    imagePaths = []
    for folder in os.listdir(training_img_path):
        folder_path = os.path.join(training_img_path, folder)
        for img in os.listdir(folder_path):
            imagePaths.append(folder_path + "/" + img)

    faceSamples = []
    Ids = []

    for imagePath in imagePaths:
        PIL_img = Image.open(imagePath).convert('L')
        img_numpy = np.array(PIL_img, 'uint8')

        Id = int(imagePath.split("_")[-2])  # extract ID
        faces = detector.detectMultiScale(img_numpy)

        for (x, y, w, h) in faces:
            faceSamples.append(img_numpy[y:y + h, x:x + w])
            Ids.append(Id)

    recognizer.train(faceSamples, np.array(Ids))
    recognizer.write(trainer_path + "/Trainer.yml")

    print("Training Complete! Trainer.yml created.")


# MAIN
print("1: Take Images")
print("2: Train Model")
choice = input("Enter your choice: ")

if choice == "1":
    take_images()
elif choice == "2":
    train_images()
else:
    print("Invalid choice!")
