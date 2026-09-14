import cv2

from src.vector_store import search_face


SFACE_MODEL = "models/face_recognition_sface_2021dec.onnx"

recognizer = cv2.FaceRecognizerSF.create(
    model = SFACE_MODEL,
    config= ""
)


def get_embedding(image, face):
    aligned_face = recognizer.alignCrop(image, face)
    embedding = recognizer.feature(aligned_face)

    return embedding


def recognize_face(image, face):
    embedding = get_embedding(image, face)

    name, score = search_face(embedding)

    threshold = 0.40

    if score < threshold:
        name = "Unknown"

    return name, score