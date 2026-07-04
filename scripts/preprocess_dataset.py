import cv2
import os

INPUT_DIR = "dataset"
OUTPUT_DIR = "processed_faces"
IMG_SIZE = 160

# Lloading Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
# Create output folder
os.makedirs(OUTPUT_DIR, exist_ok=True)

def preprocess_images():
    for person_name in os.listdir(INPUT_DIR):
        person_path = os.path.join(INPUT_DIR, person_name)
        output_person_path = os.path.join(OUTPUT_DIR, person_name)
        if not os.path.isdir(person_path):
            continue
        os.makedirs(output_person_path, exist_ok=True)
        print(f"[INFO] Processing {person_name}...")

        for img_name in os.listdir(person_path):
            img_path = os.path.join(person_path, img_name)
            try:
                img = cv2.imread(img_path)
                if img is None:
                    continue
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(
                    gray, scaleFactor=1.3, minNeighbors=5
                )
                if len(faces) == 0:
                    continue

                # Take first detected face
                (x, y, w, h) = faces[0]
                face = img[y:y+h, x:x+w]
                face = cv2.resize(face, (IMG_SIZE, IMG_SIZE))
                save_path = os.path.join(output_person_path, img_name)
                cv2.imwrite(save_path, face)

            except Exception as e:
                print(f"[ERROR] {img_path}: {e}")

    print("[INFO] Preprocessing Completed!")

if __name__ == "__main__":
    preprocess_images()