import cv2
import os

name = input("Enter your name: ").strip()
dataset_path = "dataset"
person_path = os.path.join(dataset_path, name)

if not os.path.exists(person_path):
    os.makedirs(person_path)

cap = cv2.VideoCapture(0)
count = 0

print("Starting to capture images. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    cv2.imshow("Capture Faces", frame)
    key = cv2.waitKey(1) & 0xFF

    # Save image every 10 frames approx
    if count % 10 == 0:
        img_path = os.path.join(person_path, f"{count}.jpg")
        cv2.imwrite(img_path, frame)
        print(f"Saved {img_path}")

    count += 1

    if key == ord('q') or count >= 100:  # Stop after 100 frames or pressing q
        break

cap.release()
cv2.destroyAllWindows()
print("Image capture completed.")
