import numpy as np
import cv2

cap = cv2.VideoCapture(0) # type: VideoCapture
fourcc = cv2.VideoWriter_fourcc('X','2','6','4') # type: VideoWriter_fourcc
out = cv2.VideoWriter('output.avi', fourcc, 20.0, (640, 480)) # type: VideoWriter

while(True):
    ret, frame = cap.read() # type: (bool, np.ndarray)
    if ret==False: continue
    frame2 = cv2.resize(frame, (640, 480))
    out.write(frame2)
    cv2.imshow('frame', frame2)
    if cv2.waitKey(1) & 0xFF == ord('q'): break


cap.release()
out.release()
cv2.destroyAllWindows()