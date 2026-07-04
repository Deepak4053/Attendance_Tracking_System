import cv2
import numpy as np
import pickle
import os
import time
from datetime import datetime
from keras_facenet import FaceNet

EMBEDDINGS_PATH = "embeddings/face_db.pkl"
ATTENDANCE_FILE = "attendance/attendance.csv"
print("[INFO] Loading FaceNet...")
embedder = FaceNet()
with open(EMBEDDINGS_PATH, "rb") as f:
    person_embeddings = pickle.load(f)

#  Setup
os.makedirs("attendance", exist_ok=True)
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
#  Functions
def get_embedding(face_img):
    face_img = cv2.resize(face_img, (160, 160))
    return embedder.embeddings([face_img])[0]

def recognize_face(embedding, threshold=0.9):
    min_dist = float("inf")
    name = "Unknown"
    for label, known_embedding in person_embeddings.items():
        dist = np.linalg.norm(embedding - known_embedding)
        if dist < min_dist:
            min_dist = dist
            name = label

    if min_dist > threshold:
        return "Unknown"
    return name

#  Student Attendance 
attendance_memory = {}
def mark_attendance(name):
    if name == "Unknown":
        return
    now = time.time()
    if name in attendance_memory and now - attendance_memory[name] < 10:
        return
    attendance_memory[name] = now
    with open(ATTENDANCE_FILE, "a") as f:
        f.write(f"{name},{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

# 🧠 FACE TRACKING
tracked_faces = {}
face_id_counter = 0

# 🎥 START CAMERA
cap = cv2.VideoCapture(0)
print("[INFO] Starting recognition... Press 'q' to exit")
prev_time = time.time()
while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.3, minNeighbors=5
    )

    current_time = time.time()
    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]
        center = (x + w//2, y + h//2)

        matched_id = None

        #  Match with tracked faces
        for fid, data in tracked_faces.items():
            prev_center = data["center"]
            dist = np.linalg.norm(np.array(center) - np.array(prev_center))

            if dist < 50:
                matched_id = fid
                break

        #  New face
        if matched_id is None:
            matched_id = face_id_counter
            face_id_counter += 1

        #  Recognize only once
        if matched_id not in tracked_faces:
            embedding = get_embedding(face)
            name = recognize_face(embedding)

            tracked_faces[matched_id] = {
                "name": name,
                "center": center,
                "last_seen": current_time
            }
        else:
            name = tracked_faces[matched_id]["name"]
            tracked_faces[matched_id]["center"] = center
            tracked_faces[matched_id]["last_seen"] = current_time

        #  Draw box
        color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(frame, f"{name} (ID:{matched_id})",
                    (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        mark_attendance(name)

    #  Remove old tracks
    tracked_faces = {
        fid: data for fid, data in tracked_faces.items()
        if current_time - data["last_seen"] < 2
    }
    #  Stats
    fps = 1 / (current_time - prev_time)
    prev_time = current_time
    cv2.putText(frame, f"Faces: {len(faces)}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    cv2.putText(frame, f"FPS: {int(fps)}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    cv2.imshow("Smart Face Recognition System", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("✅ System stopped.")