import cv2
import time

from recognition import get_embedding
from vector_store import add_vectors

MODEL_PATH = "models/face_detection_yunet_2023mr.onnx"

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

        detector = cv2.FaceDetectorYN.create(
            MODEL_PATH,
            "",
            (320,320),
            0.9,
            0.3,
            5000
        )

        height, width = frame.shape[:2]
        detector.setInputSize((width,height))

        _,faces = detector.detect(frame)

        if faces is not None and len(faces) > 0:

            # Use the largest detected face
            face = max(
                faces,
                key=lambda f: f[2] * f[3]
            )

            embedding = get_embedding(frame, face)

            embeddings.append(embedding)

            print(
                f"Captured sample "
                f"{len(embeddings)}/{number_of_images}"
            )

            time.sleep(1)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cv2.destroyWindow("Face Registration")

    if len(embeddings) == number_of_images:

        for embedding in embeddings:
            add_face(name, embedding)

        print(f"\n{name} successfully registered!")

        return True

    print("\nRegistration cancelled.")

    return False