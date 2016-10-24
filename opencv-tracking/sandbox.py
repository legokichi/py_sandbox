import numpy as np
import cv2
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




def gen_frames():
  cap = cv2.VideoCapture(0) # type: cv2.VideoCapture
  print("start capture")
  while True:
      ret, frame = cap.read() # type: (bool, np.ndarray)
      if not ret: break
      yield frame
  cap.release()
  print("stop capture")




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

if __name__ == "__main__":
    for f in optical_flow():
        cv2.imshow("hi", f)
        cv2.waitKey(1)
    cv2.destroyAllWindows()
