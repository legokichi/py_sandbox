import numpy as np
import cv2

cap = cv2.VideoCapture("2016-10-18-123529.mp4") # type: VideoCapture

while(True):
    ret, frame = cap.read() # type: (bool, np.ndarray)
    if ret==False:
      print("fin")
      break
    frame2 = cv2.resize(frame, (640, 480))
    cv2.imshow('frame', frame2)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()