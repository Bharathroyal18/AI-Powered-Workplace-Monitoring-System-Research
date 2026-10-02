import cv2
import numpy as np
from PIL import Image
import os
import json


# -----------------------------------------
# Paths
# -----------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")

TRAINER_FILE = os.path.join(MODELS_DIR, "trainer_face.yml")
LABEL_MAP_FILE = os.path.join(MODELS_DIR, "label_map.json")


# -----------------------------------------
# TRAIN FACE RECOGNITION MODEL
# -----------------------------------------
def train_faces(dataset_path):

    print("\n[INFO] Starting face training...\n")

    recognizer = cv2.face.LBPHFaceRecognizer_create(
        radius=1,
        neighbors=8,
        grid_x=8,
        grid_y=8
    )

    face_samples = []
    ids = []

    label_map = {}
    current_id = 0

    total_images = 0

    # Sort folder names so IDs remain stable
    people = sorted(os.listdir(dataset_path))

    for person_name in people:

        person_path = os.path.join(dataset_path, person_name)

        if not os.path.isdir(person_path):
            continue

        print(f"[INFO] Processing person: {person_name}")

        label_map[current_id] = person_name

        image_count = 0

        for image_name in os.listdir(person_path):

            if not image_name.endswith(".jpg"):
                continue

            image_path = os.path.join(person_path, image_name)

            try:
                img = Image.open(image_path).convert("L")
            except:
                print("[WARNING] Skipping corrupted image:", image_name)
                continue

            img_numpy = np.array(img, "uint8")

            # Normalize lighting
            img_numpy = cv2.equalizeHist(img_numpy)

            # Ensure same size as dataset
            img_numpy = cv2.resize(img_numpy, (220,220))

            face_samples.append(img_numpy)
            ids.append(current_id)

            image_count += 1
            total_images += 1

        print(f"   -> {image_count} images loaded")

        # Warn if too few images
        if image_count < 15:
            print("[WARNING] Very few images for this person.")

        current_id += 1

    if len(face_samples) == 0:
        print("[ERROR] No training images found.")
        return

    print("\n[INFO] Training LBPH model...")

    recognizer.train(face_samples, np.array(ids))

    os.makedirs(MODELS_DIR, exist_ok=True)

    recognizer.write(TRAINER_FILE)

    # Save label mapping for recognizer
    with open(LABEL_MAP_FILE,"w") as f:
        json.dump(label_map,f)

    print("\n================================")
    print("[INFO] Training Complete")
    print(f"[INFO] Total People: {len(label_map)}")
    print(f"[INFO] Total Images: {total_images}")
    print(f"[INFO] Model saved: {TRAINER_FILE}")
    print(f"[INFO] Labels saved: {LABEL_MAP_FILE}")
    print("================================\n")

    return label_map


# -----------------------------------------
# MAIN
# -----------------------------------------
if __name__ == "__main__":

    dataset_path = os.path.join(BASE_DIR, "dataset", "faces")

    if not os.path.exists(dataset_path):
        print("[ERROR] Dataset folder not found.")
        exit()

    train_faces(dataset_path)

    print("[INFO] Training finished successfully.")