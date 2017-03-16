import numpy as np
import cv2
import time

def resize(ratio: float, frame):
  size = (int(frame.shape[1]*ratio), int(frame.shape[0]*ratio))
  return cv2.resize(frame, size)

def gen_frames():
  cap = cv2.VideoCapture('80eab061-a9b2-49cb-8996-88783a7bc0f8.addSeekHeadAndReplaceDuration.webm') # type: cv2.VideoCapture
  #cap = cv2.VideoCapture(0) # type: cv2.VideoCapture
  print("start capture")
  while True:
      ret, frame = cap.read() # type: (bool, np.ndarray)
      if not ret: break
      yield resize(1/8, frame)
  cap.release()
  print("stop capture")

# https://github.com/icoxfog417/cv_tutorial/blob/master/opticalflow/cv_opticalflow_tutorial.ipynb
# http://whoopsidaisies.hatenablog.com/entry/2013/12/15/020420
def optical_flow_dense():
    to_grayscale = lambda f: cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
    params = dict(
        pyr_scale=0.5,
        levels=1,
        winsize=30,
        iterations=1,
        poly_n=5,
        poly_sigma=1.1,
        flags=0
    )

    frames = gen_frames()
    _frame = next(frames) # first frame
    p_frame = to_grayscale(_frame)
    hsv = np.zeros_like(_frame) # (width, heitht, (hsv_h, hsv_s))
    hsv[...,0] = 255 # hsv_h = 255
    hsv[...,1] = 255 # hsv_s = 255
    hsv[...,2] = 255 # hsv_s = 255

    for frame in frames:
        cv2.imshow("origin", frame)
        
        glayed = to_grayscale(frame)
        
        # calculate optical flow
        flow = cv2.calcOpticalFlowFarneback(p_frame, glayed, None, **params) # type: numpy.ndarray
        if flow is None:
            print("none")
            continue
        # flow[...,0], flow[...,1] は 各画素の mv の (x,y)
        _abs, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
        hsv[...,0] = cv2.normalize(_abs, None, 0, 255, cv2.NORM_MINMAX)*0.5
        
        
        #hsv[...,2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)  # magnitude to 0-255 scale
        frame = cv2.cvtColor(hsv,cv2.COLOR_HSV2RGB)

        p_frame = glayed.copy()
        yield frame

if __name__ == "__main__":
    prev = time.time()
    i = 0
    for f in optical_flow_dense():
        cv2.imshow("hi", f)
        cv2.waitKey(1)
        i += 1
        if(i > 30):
          now = time.time()
          print(i/(now - prev))
          i = 0
          prev = now
        
    cv2.destroyAllWindows()

# usage
# env PYTHONPATH=/usr/local/Cellar/opencv3/3.1.0_4/lib/python3.5/site-packages/:$PYTHONPATH python  