import cv2

from src.registration import register_person


cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("Could not open camera.")
    exit()


name = input("Enter person's name: ")

register_person(
    cap,
    name,
    number_of_images=3
)

cap.release()
cv2.destroyAllWindows()