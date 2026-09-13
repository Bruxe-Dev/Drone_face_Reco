import cv2
import pickle as pc 
import numpy as np 

YUNET_PATH ="models/face_detection_yunet_2023mar.onnx"
SFACE_PATH ="models/face_recognition_sface_2021dec.onnx"

ENCODING_PATH ="encodings/faces.pkl"

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

with open(ENCODING_PATH, "rb") as file:
    known_faces = pc.load(file)

def cosine_similarity(a,b):

    a = a.flatten()
    b = b.flatten()

    return np.dot(a,b) / (
        np.linalg.norm(a) * np.linalg.norm(b)
    )

def recognize_face(image,face):

    align_face = recognizer.alignCrop(
        image,
        face
    )

    embedding = recognizer.feature(
        aligned_face
    )

    best_name = "Unknown"
    best_score = -1

    for known_face in known_faces:

        score = cosine_similarity(
            embedding,
            known_face["embedding"]
        )

        if score > best_score:
            best_score = score
            best_name = known_face["name"]

    # Recognition threshold
    threshold = 0.40

    if best_score < threshold:
        best_name = "Unknown"


    return best_name,best_score