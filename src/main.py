import cv2

from detection import detector
from recognition import recognize_face
from speech import greet

camera = cv2.VideoCapture(1, cv2.CAP_DSHOW)

if not camera.isOpened():
    print("Could not access camera")
    exit()

print("Camera started!")
print("Press Q to quit.")

greeted_people = set()

STABILITY_FRAMES = 15

stable_name = None
stable_frames = 0


while True:

    success, frame = camera.read()

    if not success:
        print("Could not read frame")
        break

    frame = cv2.flip(frame, 1)

    height, width = frame.shape[:2]

    # Camera center
    frame_center_x = width / 2
    frame_center_y = height / 2

    detector.setInputSize((width, height))

    _, faces = detector.detect(frame)


    target = None
    target_name = "Unknown"
    target_score = 0

    if faces is not None:

        best_distance = float("inf")

        for face in faces:

            x, y, w, h = face[:4]

            x = int(x)
            y = int(y)
            w = int(w)
            h = int(h)

            # Face center
            face_center_x = x + w / 2
            face_center_y = y + h / 2

            # Distance from camera center
            distance = (
                (face_center_x - frame_center_x) ** 2
                +
                (face_center_y - frame_center_y) ** 2
            ) ** 0.5

            center_tolerance_x = width * 0.20
            center_tolerance_y = height * 0.25

            centered = (
                abs(face_center_x - frame_center_x)
                < center_tolerance_x
                and
                abs(face_center_y - frame_center_y)
                < center_tolerance_y
            )

            minimum_face_width = width * 0.12

            close_enough = w >= minimum_face_width


            if centered and close_enough:

                if distance < best_distance:

                    best_distance = distance
                    target = face


    if target is not None:

        target_name, target_score = recognize_face(
            frame,
            target
        )

        x, y, w, h = target[:4]

        x = int(x)
        y = int(y)
        w = int(w)
        h = int(h)

        # Draw target box
        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )

        label = f"{target_name} ({target_score:.2f})"

        cv2.putText(
            frame,
            label,
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

        if target_name != "Unknown":

            if target_name == stable_name:

                stable_frames += 1

            else:

                stable_name = target_name
                stable_frames = 1


            # Show stability progress
            cv2.putText(
                frame,
                f"Confirming: {stable_frames}/{STABILITY_FRAMES}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2
            )

            if (
                stable_frames >= STABILITY_FRAMES
                and target_name not in greeted_people
            ):

                greet(target_name)

                greeted_people.add(target_name)

                stable_frames = 0


        else:

            stable_name = None
            stable_frames = 0

    else:

        # No suitable target
        stable_name = None
        stable_frames = 0

    cv2.imshow(
        "Drone Face Recognition",
        frame
    )

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


camera.release()
cv2.destroyAllWindows()