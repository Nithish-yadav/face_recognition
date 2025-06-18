import cv2
import os
import numpy as np
import pandas as pd
import pandas.errors
from datetime import datetime

print("Current working directory:", os.getcwd())

# Load the trained model
recognizer = cv2.face.LBPHFaceRecognizer_create(radius=2, neighbors=10, grid_x=8, grid_y=8)
recognizer.read("trainer.yml")

# Load label map
with open("label_map.txt", "r") as f:
    labels = f.read().splitlines()

# Attendance data frame (empty initially, to track current session attendance)
attendance_df = pd.DataFrame(columns=["Name", "Date", "Time"])

# Function to mark attendance, saving daily CSV files
def mark_attendance(name):
    global attendance_df
    now = datetime.now()
    date_string = now.strftime("%Y-%m-%d")
    time_string = now.strftime("%H:%M:%S")

    # Daily attendance filename
    daily_file = f"attendance_{date_string}.csv"

    # Check if already marked in this session (avoid duplicates)
    if not ((attendance_df['Name'] == name) & (attendance_df['Date'] == date_string)).any():
        new_entry = {"Name": name, "Date": date_string, "Time": time_string}
        attendance_df.loc[len(attendance_df)] = new_entry

        # Append to daily CSV (create if not exists)
        if os.path.exists(daily_file):
            daily_df = pd.read_csv(daily_file)
            # Prevent duplicates in file as well
            if not ((daily_df['Name'] == name) & (daily_df['Date'] == date_string)).any():
                daily_df = pd.concat([daily_df, pd.DataFrame([new_entry])], ignore_index=True)
                daily_df.to_csv(daily_file, index=False)
        else:
            pd.DataFrame([new_entry]).to_csv(daily_file, index=False)

        print(f"Attendance marked for {name} at {time_string}")

# Initialize webcam and face detector
cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

print("Starting attendance recognition. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)

    recognized_names = []

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        id_, confidence = recognizer.predict(roi_gray)

        if confidence < 70:  # Adjust threshold for accuracy
            name = labels[id_]
            mark_attendance(name)
            recognized_names.append(name)
            cv2.putText(frame, name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
            cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
        else:
            cv2.putText(frame, "Unknown", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
            cv2.rectangle(frame, (x,y), (x+w,y+h), (0,0,255), 2)

    # Optionally show all recognized names at top-left corner
    if recognized_names:
        names_text = " | ".join(set(recognized_names))
        cv2.putText(frame, f"Recognized: {names_text}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,0), 2)

    cv2.imshow("Attendance", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
