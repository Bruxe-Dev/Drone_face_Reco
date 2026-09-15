import cv2
import time

from src.detection import detect_faces
from src.recognition import get_embedding
from src.vector_store import add_face


def register_person(cap, name, number_of_images=3):

    embeddings = []

    print(f"\nRegistering {name}...")
    print(f"Capturing {number_of_images} face samples.")
    print("Look directly at the camera.\n")

    while len(embeddings) < number_of_images:

        success, frame = cap.read()

        if not success:
            print("Could not read camera frame.")
            continue

        frame = cv2.flip(frame, 1)

        faces = detect_faces(frame)

        cv2.putText(
            frame,
            f"Registering: {name}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            f"Samples: {len(embeddings)}/{number_of_images}",
            (20, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

        if faces is not None and len(faces) > 0:

            # Pick largest face
            face = max(
                faces,
                key=lambda f: f[2] * f[3]
            )

            x, y, w, h = face[:4]

            cv2.rectangle(
                frame,
                (x, y),
                (x + w, y + h),
                (0, 255, 0),
                2
            )

            embedding = get_embedding(
                frame,
                face
            )

            embeddings.append(embedding)

            print(
                f"Captured sample "
                f"{len(embeddings)}/{number_of_images}"
            )

            time.sleep(1)

        cv2.imshow(
            "Face Registration",
            frame
        )

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cv2.destroyWindow("Face Registration")

    if len(embeddings) == number_of_images:

        for embedding in embeddings:

            add_face(
                name,
                embedding
            )

        print(
            f"\n{name} successfully registered!"
        )

        return True

    print("\nRegistration cancelled.")

    return False