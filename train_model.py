import cv2
import os
import numpy as np

dataset_path = "dataset"

recognizer = cv2.face.LBPHFaceRecognizer_create()
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

faces = []
labels = []
label_ids = {}
current_id = 0

for root, dirs, files in os.walk(dataset_path):
    for dir_name in dirs:
        label = dir_name
        if label not in label_ids:
            label_ids[label] = current_id
            current_id += 1
        id_ = label_ids[label]

        person_folder = os.path.join(root, dir_name)
        for img_name in os.listdir(person_folder):
            if img_name.lower().endswith(("png", "jpg", "jpeg")):
                img_path = os.path.join(person_folder, img_name)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    continue
                faces_rects = face_cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=5)
                for (x, y, w, h) in faces_rects:
                    faces.append(img[y:y+h, x:x+w])
                    labels.append(id_)

recognizer.train(faces, np.array(labels))
recognizer.save("trainer.yml")

with open("label_map.txt", "w") as f:
    for label, id_ in label_ids.items():
        f.write(f"{id_}:{label}\n")

print("Training complete. Model saved as 'trainer.yml'.")
