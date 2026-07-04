import os
import cv2
import numpy as np
import pickle
from keras_facenet import FaceNet

DATASET_DIR = "processed_faces"
OUTPUT_PATH = "embeddings/face_db.pkl"
print("[INFO] Loading FaceNet...")
embedder = FaceNet()

# Dictionary to hold person embeddings
person_embeddings = {}

# Generate embeddings for each person in the dataset
for person_name in os.listdir(DATASET_DIR):
    person_path = os.path.join(DATASET_DIR, person_name)
    if not os.path.isdir(person_path):
        continue

    print(f"[INFO] Processing {person_name}...")
    embeddings_list = []
    for img_name in os.listdir(person_path):
        img_path = os.path.join(person_path, img_name)

        img = cv2.imread(img_path)
        if img is None:
            continue
        img = cv2.resize(img, (160, 160))

        # Generate embedding
        emb = embedder.embeddings([img])[0]
        embeddings_list.append(emb)

    #  Take average embedding 
    if len(embeddings_list) > 0:
        mean_embedding = np.mean(embeddings_list, axis=0)
        person_embeddings[person_name] = mean_embedding

# Saving embeddings to file
os.makedirs("embeddings", exist_ok=True)
with open(OUTPUT_PATH, "wb") as f:
    pickle.dump(person_embeddings, f)
print("✅ Face database created successfully!")