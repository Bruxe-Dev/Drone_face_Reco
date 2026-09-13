import cv2
from recognition import recognize_face


camera = cv2.VideoCapture(1, cv2.CAP_DSHOW)

if not camera.isOpened():
    print("Could not access camera")
    exit()


print("Camera started!")
print("Press Q to quit.")


while True:

    success, frame = camera.read()

    if not success:
        print("Could not read frame")
        break

    frame = cv2.flip(frame, 1)

    height, width = frame.shape[:2]

    from detection import detector

    detector.setInputSize((width, height))

    _, faces = detector.detect(frame)

    if faces is not None:

        for face in faces:

            name, score = recognize_face(
                frame,
                face
            )

            x, y, w, h = face[:4]

            x = int(x)
            y = int(y)
            w = int(w)
            h = int(h)

            cv2.rectangle(
                frame,
                (x, y),
                (x + w, y + h),
                (0, 255, 0),
                2
            )
            label = f"{name} ({score:.2f})"

            cv2.putText(
                frame,
                label,
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
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