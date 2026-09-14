import cv2
import time

from recognition import get_embedding
from vector_store import add_vectors

def register_person(cap,name,number_of_images=3):
    embedding = []

    print(f"\nRegistering {name}...")
    print(f"Capturing {number_of_images} face samples.")
    print("Look directly at the camera.\n")

    while len(embedding) < number_of_images:
        success, frame = cap.read()

        if not success:
            print("Could not read Camera Frame")
            continue

        frame = cv2.flip(frame,1)

        cv2.putText(
            frame,
            f"Registering {name}",
            (20,40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0,255,255),
            2
        )

        cv2.imshow(
            "Face Recognition",frame
        )