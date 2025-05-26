import cv2
import numpy as np
import face_recognition
import tkinter as tk
from tkinter import ttk, messagebox 
import os
from PIL import Image, ImageTk
import pandas as pd
import csv

dataPath = r"C:/Foto/FaceRecognitionSystem/data"
databaseFile = r"C:/Foto/FaceRecognitionSystem/database.csv"

register = False
recognizeFrame = False
sampleNum = 20
name = ""
Id = ""
status = ""

faces = []
Ids = []

def createImages(frame, count):
    global dataPath, name, Id
    if not name or not Id:
        return frame
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_frame)
    for (top, right, bottom, left) in face_locations:
        face_image = rgb_frame[top:bottom, left:right]
        pil_image = Image.fromarray(face_image)
        pil_image.save(os.path.join(dataPath, f"{name}.{Id}.{count}.jpg"))
        cv2.rectangle(frame, (left, top), (right, bottom), (255, 0, 0), 2)
    return frame

def writeDatabase(databaseFile, row):
    if not os.path.exists(databaseFile):
        with open(databaseFile, 'a+') as csvFile:
            writer = csv.writer(csvFile)
            writer.writerow(["Id", "Name", "Status"])
    with open(databaseFile, 'a+') as csvFile:
        writer = csv.writer(csvFile)
        writer.writerow(row)
    return "Saved to database"

def getImagesAndLabels(path):
    imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
    faces = []
    Ids = []
    for imagePath in imagePaths:
        extension = os.path.splitext(imagePath)[1]
        if extension != '.jpg':
            continue
        image = face_recognition.load_image_file(imagePath)
        face_encodings = face_recognition.face_encodings(image)
        if face_encodings:
            face_encoding = face_encodings[0]
            Id = int(os.path.split(imagePath)[-1].split(".")[1])
            faces.append(face_encoding)
            Ids.append(Id)
    return faces, Ids

def Train(path):
    faces, Ids = getImagesAndLabels(path)
    data = {"faces": faces, "Ids": Ids}
    project_dir = os.path.dirname(os.path.abspath(__file__))  # Абсолютный путь к текущему файлу
    save_path = os.path.join(project_dir, "train_data.npy")
    np.save(save_path, data)
    return "Training finished..."

def TrackImages(frame):
    project_dir = os.path.dirname(os.path.abspath(__file__))
    save_path = os.path.join(project_dir, "train_data.npy")
    if not os.path.exists(save_path):
        return frame, "No training data available"
    data = np.load(save_path, allow_pickle=True).item()
    known_face_encodings = data["faces"]
    known_face_ids = data["Ids"]
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
    df = pd.read_csv(databaseFile, delimiter=',')
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            Id = known_face_ids[best_match_index]
            person = df.loc[df['Id'] == Id]['Name'].values[0]
            status = df.loc[df['Id'] == Id]['Status'].values[0]
            person_info = f"{Id}-{person}-{status} {int((1 - face_distances[best_match_index]) * 100)}%"
        else:
            person_info = "Unknown"
        cv2.rectangle(frame, (left, top), (right, bottom), (255, 0, 0), 2)
        cv2.putText(frame, person_info, (left, bottom + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return frame, "Recognition complete"

count = 0

def main():
    global register, sampleNum, dataPath, name, Id, status, recognizeFrame
    root = tk.Tk()
    root.title("Face Recognition System")
    root.configure(bg="#ccffcc")  # Цвет фона светло-салатовый

    style = ttk.Style()
    style.configure("TButton", font=("Helvetica", 12), background="#007ACC", foreground="blue")
    style.map("TButton", background=[('active', '#005C99')], foreground=[('active', 'blue')])

    def register_action():
        global register, count
        register = True
        count = 0

    def train_action():
        info = Train(dataPath)
        messagebox.showinfo("Info", info)

    def recognize_action():
        global recognizeFrame
        recognizeFrame = True

    left_panel = tk.Frame(root, bg="#ccffcc")
    left_panel.grid(row=0, column=0, padx=10, pady=10)
    title_label = ttk.Label(left_panel, text="Face Recognition", font=("Helvetica", 20), background="#ccffcc")
    title_label.grid(row=0, column=0)
    image_label = ttk.Label(left_panel)
    image_label.grid(row=1, column=0)

    right_panel = tk.Frame(root, bg="#ccffcc", padx=10, pady=10)
    right_panel.grid(row=0, column=1, padx=10, pady=10, sticky="n")
    id_label = ttk.Label(right_panel, text="Id:",font=("Helvetica", 11), background="#ccffcc")
    id_label.grid(row=0, column=0, pady=5)
    id_entry = ttk.Entry(right_panel)
    id_entry.grid(row=0, column=1, pady=5)
    name_label = ttk.Label(right_panel, text="Name:",font=("Helvetica", 11), background="#ccffcc")
    name_label.grid(row=1, column=0, pady=5)
    name_entry = ttk.Entry(right_panel)
    name_entry.grid(row=1, column=1, pady=5)
    status_label = ttk.Label(right_panel, text="Status:",font=("Helvetica", 11), background="#ccffcc")
    status_label.grid(row=2, column=0, pady=5)
    status_entry = ttk.Entry(right_panel)
    status_entry.grid(row=2, column=1, pady=5)

    register_button = ttk.Button(right_panel, text="1. Register", command=register_action)
    register_button.grid(row=3, column=1, columnspan=2, pady=10)
    train_button = ttk.Button(right_panel, text="2. Train", command=train_action)
    train_button.grid(row=4, column=1, columnspan=2, pady=10)
    recognize_button = ttk.Button(right_panel, text="3. Recognize", command=recognize_action)
    recognize_button.grid(row=5, column=1, columnspan=2, pady=10)

    cap = cv2.VideoCapture(0)

    def update_frame():
        global register, recognizeFrame, count, name, Id, status
        ret, frame = cap.read()
        if not ret:
            return
        
        name = name_entry.get()
        Id = id_entry.get()
        status = status_entry.get()

        if register:
            frame = createImages(frame, count)
            info = "Saving " + str(count)
            count += 1
            if count > sampleNum:
                row = [Id, name, status]
                info = writeDatabase(databaseFile, row)
                register = False
                messagebox.showinfo("Info", info)
        
        if recognizeFrame:
            frame, info = TrackImages(frame)
            title_label.config(text=info)
        
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        imgtk = ImageTk.PhotoImage(image=img)
        image_label.imgtk = imgtk
        image_label.configure(image=imgtk)
        root.after(20, update_frame)

    root.after(0, update_frame)
    root.mainloop()

    cap.release()
    cv2.destroyAllWindows()

main()
