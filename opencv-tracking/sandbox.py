import numpy as np
import cv2
import time
# http://qiita.com/icoxfog417/items/357e6e495b7a40da14d8#%E5%AE%9F%E8%B7%B5%E7%B7%A8
# https://github.com/icoxfog417/cv_tutorial/tree/master/opticalflow
# https://github.com/icoxfog417/cv_tutorial/blob/master/opticalflow/cv_opticalflow_tutorial.ipynb
# goodFeaturesToTrack
# calcOpticalFlowPyrLK

# params for ShiTomasi corner detection
FEATURE_COUNT = 100
FEATURE_PARAMS = dict(
    maxCorners=FEATURE_COUNT,
    qualityLevel=0.3,
    minDistance=7,
    blockSize=7
)

# Parameters for lucas kanade optical flow
LK_PARAMS = dict(
    winSize  = (15,15),
    maxLevel = 2,
    criteria = (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 10, 0.03)
)

# color for drawing (create FEATURE_COUNT colors each of these is RGB color)
COLOR = np.random.randint(0, 255,(FEATURE_COUNT, 3))

def resize(ratio:float, frame):
  size = (int(frame.shape[1]*ratio), int(frame.shape[0]*ratio))
  return cv2.resize(frame, size)


def gen_frames():
  cap = cv2.VideoCapture(0) # type: cv2.VideoCapture
  print("start capture")
  while True:
      ret, frame = cap.read() # type: (bool, np.ndarray)
      if not ret: break
      yield resize(1/2, frame)
  cap.release()
  print("stop capture")


# https://github.com/icoxfog417/cv_tutorial/blob/master/opticalflow/cv_opticalflow_tutorial.ipynb
def optical_flow():
    get_features = lambda f: cv2.goodFeaturesToTrack(f, mask=None, **FEATURE_PARAMS)
    to_grayscale = lambda f: cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
    frames = gen_frames()
    p_frame = to_grayscale(next(frames))
    p_features = get_features(p_frame)
    
    for frame in frames:
        glayed = to_grayscale(frame)
        # http://opencv.jp/opencv-2svn/cpp/motion_analysis_and_object_tracking.html#cv-calcopticalflowpyrlk
        c_features, st, err = cv2.calcOpticalFlowPyrLK(p_frame, glayed, p_features, None, **LK_PARAMS) # type: <class 'numpy.ndarray'> <class 'numpy.ndarray'> <class 'numpy.ndarray'>  
        # select good points (st = 1 if tracking is successed)
        tracked = c_features[st==1]

        # draw line
        for i, current in enumerate(tracked):
            x1, y1 = current.ravel()
            frame = cv2.circle(frame, (x1, y1), 5, COLOR[i].tolist(), -1)
        p_frame = glayed.copy()
        p_features = c_features.reshape(-1, 1, 2)
        yield frame


# https://github.com/icoxfog417/cv_tutorial/blob/master/opticalflow/cv_opticalflow_tutorial.ipynb
# http://whoopsidaisies.hatenablog.com/entry/2013/12/15/020420
def optical_flow_dense():
    to_grayscale = lambda f: cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
    params = dict(
        pyr_scale=0.5,
        levels=3,
        winsize=15,
        iterations=3,
        poly_n=5,
        poly_sigma=1.1,
        flags=0
    )

    frames = gen_frames()
    _frame = next(frames)
    p_frame = to_grayscale(_frame)
    hsv = np.zeros_like(_frame)
    hsv[...,1] = 255

    for frame in frames:
        glayed = to_grayscale(frame)
        # calculate optical flow
        flow = cv2.calcOpticalFlowFarneback(p_frame, glayed, None, **params) # type: np.ndarray?
        if flow is None:
            print("none")
            continue
        # optical flow's magnitudes and angles
        mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
        hsv[...,0] = ang*180/np.pi/2
        hsv[...,2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)  # magnitude to 0-255 scale
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
