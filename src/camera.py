import cv2 

capture = cv2.VideoCapture(1)

while True:
    sucess, frame = capture.read()

    if not sucess:
        print("Could not access Camera")
        break

    frame = cv2.flip(frame,1)
    cv2.imshow("DFR",frame)

    if(cv2.waitKey(1) & 0xFF == ord('q')):
        break
    
capture.release()
cv2.destroyAllWindows