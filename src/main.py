import cv2
from detection import detect_faces


camera = cv2.VideoCapture(1)

while True:
    success, frame = camera.read()

    if not success:
        print("Camera Access Failed!")
        break

    frame = cv2.flip(frame, 1)

    faces = detect_faces(frame)

    if faces is not None:
        for face in faces:
            x, y, w, h = face[:4]

            x = int(x)
            y = int(y)
            w = int(w)
            h = int(h)

            cv2.rectangle(
                frame,
                (x, y),
                (x + w, y + h),
                (255, 0, 0),
                2
            )

    cv2.imshow("DFR", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


camera.release()
cv2.destroyAllWindows()