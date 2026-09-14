import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

import cv2

from src.detection import detect_faces
from src.recognition import recognize_face
from src.registration import register_person
from src.speech import greet


STABILITY_FRAMES = 15

greeted_people = set()

stable_name = None
stable_frames = 0


camera = cv2.VideoCapture(1, cv2.CAP_DSHOW)

if not camera.isOpened():
    print("Could not open camera.")
    exit()


while True:

    success, frame = camera.read()

    if not success:
        print("Could not read camera frame.")
        break

    frame = cv2.flip(frame, 1)

    height, width = frame.shape[:2]

    faces = detect_faces(frame)

    target_face = None

    valid_faces = []

    for face in faces if faces is not None else []:

        x, y, w, h = face[:4]

        face_center_x = x + w / 2
        face_center_y = y + h / 2

        frame_center_x = width / 2
        frame_center_y = height / 2

        center_x_limit = width * 0.20
        center_y_limit = height * 0.25

        minimum_face_width = width * 0.12

        centered = (
            abs(face_center_x - frame_center_x)
            < center_x_limit
            and
            abs(face_center_y - frame_center_y)
            < center_y_limit
        )

        close_enough = w >= minimum_face_width

        if centered and close_enough:
            valid_faces.append(face)

    if valid_faces:

        target_face = min(
            valid_faces,
            key=lambda face: (
                abs((face[0] + face[2] / 2) - width / 2)
                +
                abs((face[1] + face[3] / 2) - height / 2)
            )
        )

        x, y, w, h = (int(v) for v in target_face[:4])

        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )

        name, score = recognize_face(
            frame,
            target_face
        )

        cv2.putText(
            frame,
            f"{name} ({score:.2f})",
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

        if name == stable_name:
            stable_frames += 1
        else:
            stable_name = name
            stable_frames = 1

        if stable_frames >= STABILITY_FRAMES:

            if name != "Unknown":

                if name not in greeted_people:

                    greet(name)

                    greeted_people.add(name)

                stable_frames = 0

            else:

                print("\nUnknown person detected.")

                print(
                    "Please look at the camera "
                    "while registering."
                )

                new_name = input(
                    "Enter person's name: "
                ).strip()

                if new_name:

                    registered = register_person(
                        camera,
                        new_name,
                        number_of_images=3
                    )

                    if registered:

                        greet(new_name)

                        greeted_people.add(
                            new_name
                        )

                stable_name = None
                stable_frames = 0

    else:

        stable_name = None
        stable_frames = 0

    cv2.putText(
        frame,
        "Press Q to quit",
        (20, height - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    cv2.imshow(
        "Drone Face Recognition",
        frame
    )

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


camera.release()
cv2.destroyAllWindows()