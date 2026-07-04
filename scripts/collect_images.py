import cv2
import os
import time

DATASET_PATH = "dataset"
PERSON_NAME = input("Enter person name: ").strip()
NUM_IMAGES = 20
DELAY = 1.5

# Create folder
person_path = os.path.join(DATASET_PATH, PERSON_NAME)
os.makedirs(person_path, exist_ok=True)

# Load face detector
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
cap = cv2.VideoCapture(0)
count = 0
last_capture_time = time.time()
print("[INFO] Look at camera and change poses...")
while True:
    ret, frame = cap.read()
    if not ret:
        break

    display_frame = frame.copy()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.3, minNeighbors=5
    )
    current_time = time.time()
    for (x, y, w, h) in faces:
        cv2.rectangle(display_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Capture with delay
        if current_time - last_capture_time > DELAY and count < NUM_IMAGES:
            face = frame[y:y+h, x:x+w]
            img_name = os.path.join(person_path, f"{count}.jpg")
            cv2.imwrite(img_name, face)
            print(f"[INFO] Captured image {count}")
            count += 1
            last_capture_time = current_time

    # Show instructions
    cv2.putText(display_frame, f"Images: {count}/{NUM_IMAGES}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                1, (0, 255, 0), 2)

    cv2.imshow("Collecting Faces", display_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    if count >= NUM_IMAGES:
        break

cap.release()
cv2.destroyAllWindows()
print(f"[INFO] Done! Collected {count} images.")