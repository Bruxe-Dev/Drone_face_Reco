import cv2
import os
import time

def enhance_frame(frame):
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)

    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    l = clahe.apply(l)

    enhanced = cv2.merge((l, a, b))

    return cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)

name = input("Enter person's name: ")

output_folder = os.path.join("data", name)

if not os.path.exists(output_folder):
    os.makedirs(output_folder)
    print(f"Created folder: {output_folder}")

cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("Error: Could not open webcam")
    exit()

image_count = len(os.listdir(output_folder))

print(f"\nCollecting images for: {name}")
print("Images will be captured every 5 seconds.")
print("Press Q to quit.\n")

last_capture_time = time.time()

while True:

    ret, frame = cap.read()

    if not ret:
        print("Error: Failed to capture image")
        break

    frame = cv2.flip(frame, 1)
    frame = enhance_frame(frame)

    cv2.putText(
        frame,
        f"Person: {name}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.imshow("Dataset Capture - Press Q to quit", frame)

    current_time = time.time()

    if current_time - last_capture_time >= 5:

        image_count += 1

        filename = f"{image_count:02d}.jpg"
        filepath = os.path.join(output_folder, filename)

        cv2.imwrite(filepath, frame)

        print(f"Captured: {filepath}")

        last_capture_time = current_time

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

print(f"\nFinished collecting images for {name}")
print(f"Total images: {image_count}")
print(f"Saved in: {os.path.abspath(output_folder)}")