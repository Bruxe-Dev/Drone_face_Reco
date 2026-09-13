import os 
import numpy as np 
import pickle as pc 
import cv2

YUNET_PATH ="models/face_detection_yunet_2023mar.onnx"
SFACE_PATH ="models/face_recognition_sface_2021dec.onnx"

DATASET_PATH ="data"
OUTPUT_PATH ="encodings/faces.pkl"

detector = cv2.FaceDetectorYN.create(
    model= YUNET_PATH,
    config= "",
    input_size= (320,320),
    score_threshold= 0.9,
    nms_threshold= 0.3,
    top_k= 500
)

recognizer = cv2.FaceRecognizerSF.create(
    model= SFACE_PATH,
    config= ""
)

known_faces = []

for person_name in os.listdir(DATASET_PATH):
    person_folder = os.path.join(DATASET_PATH,person_name)

    if not os.path.isdir(person_folder):
        continue

    print(f"Processing: {person_name}")

    for image_name in os.listdir(person_folder):
        image_path = os.path.join(person_folder,image_name)

        image = cv2.imread(image_path)

        if image is None:
            print(f"Couldn't read: {image_path}")
            continue
        
        h,w = image.shape[:2]

        detector.setInputSize((w,h))

        _,faces = detector.detect(image)

        if faces is None or len(faces) == 0:
            print(f"No faces Detected: {image_path}")
            continue
        
        face = faces[0]

        align_face = recognizer.alignCrop(
            image,
            face
        )

        embeddings = recognizer.feature(
            align_face
        )

        known_faces.append({
            "name":person_name,
            "embeddings":embeddings
        })

        print(f"Encoded: {image_path}")

os.makedirs("encodings", exist_ok=True)

with open(OUTPUT_PATH, "wb") as file:
    pc.dump(known_faces, file)


print("\n===========================")
print("Encoding complete!")
print(f"Total embeddings: {len(known_faces)}")
print(f"Saved to: {OUTPUT_PATH}")
print("==============================")