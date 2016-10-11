# http://qiita.com/Kazuhito/items/cccdeb55d101dcc69203
# UI キット Tinker を削除
import numpy as np
import cv2

# パターン
chessboard = "../Webcam/2016-10-11-154033.jpg"
# chessboard = "../Webcam/2016-10-11-154150.jpg"
# chessboard = "../Webcam/2016-10-11-154140.jpg"
square_side_length = 24.0 # チェスボード内の正方形の1辺のサイズ(mm)
grid_intersection_size = (10, 7) # チェスボード内の格子数

pattern_points = np.zeros( (np.prod(grid_intersection_size), 3), np.float32 )
pattern_points[:,:2] = np.indices(grid_intersection_size).T.reshape(-1, 2)
pattern_points *= square_side_length
object_points = []
image_points = []

camera_mat, dist_coef = [], []

# 読み込み
frame = cv2.imread(chessboard, cv2.IMREAD_GRAYSCALE)

# コーナー検出
found, corners = cv2.findChessboardCorners(frame, grid_intersection_size)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001) # termination criteria
cv2.cornerSubPix(frame, corners, grid_intersection_size, (-1,-1), criteria)
# 現在のOpenCVではfindChessboardCorners()内で、cornerSubPix()相当の処理が実施されている？要確認

'''
cv2.putText(frame, "Enter:Capture Chessboard(" + str(capture_count) + ")", (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,255), 1)
cv2.putText(frame, "N    :Completes Calibration Photographing", (100, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,255), 1)
cv2.putText(frame, "ESC  :terminate program", (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,255), 1)
'''

# 描画
cv2.drawChessboardCorners(frame, grid_intersection_size, corners, found)


# チェスボードコーナー検出情報を追加
image_points.append(corners)
object_points.append(pattern_points)
'''
std::vector<std::vector<cv::Point3f>> object_points;
  InputArrayOfArrays objectPoints,
    // vector of vectors of calibration pattern points in the calibration pattern coordinate space.
    // object_pointsにチェスボードのコーナー座標（三次元）を各画像ごとに入れる。
    // 通常は世界座標系の原点をチェスボード上のどこかに固定してZ軸をボードの法線方向にするため、どの画像にも同一のコーナー座標群が入る。
std::vector<std::vector<cv::Point2f>> image_points;
  InputArrayOfArrays imagePoints,
    // vector of vectors of the projections of calibration pattern points. imagePoints.size() and objectPoints.size() and imagePoints[i].size() must be equal to objectPoints[i].size() for each i.
    // findChessboardCorners()で検出した画像上のコーナー座標（サブピクセル精度）を入れる。
	  // コーナーの並び順はobject_pointsと対応している必要
cv::Size img_size;
  const Size& image_size,
    // Size of the image used only to initialize the intrinsic camera matrix.
    // チェスボードを撮影した画像のサイズ
'''
print(type(object_points), len(object_points), type(object_points[0]), object_points[0].shape) # <class 'list'> 1 <class 'numpy.ndarray'> (70, 3)
print(type(image_points),  len(image_points),  type(image_points[0] ), image_points[0].shape ) # <class 'list'> 1 <class 'numpy.ndarray'> (70, 1, 2)
# チェスボード撮影を終了し、カメラ内部パラメータを求めます
retval, K, D, rvecs, tvecs = cv2.fisheye.calibrate(object_points, image_points, image_size=(frame.shape[1],frame.shape[0]), K=None, D=None)
# https://github.com/opencv/opencv/blob/master/modules/calib3d/src/fisheye.cpp#L720
# OpenCV Error: Assertion failed
#   (objectPoints.type() == CV_32FC3 || objectPoints.type() == CV_64FC3) in calibrate,
#   file /tmp/opencv3-20161008-27623-6s9pji/opencv-3.1.0/modules/calib3d/src/fisheye.cpp, line 695
# #define CV_32FC3 CV_MAKETYPE(CV_32F,3)
# #define CV_64FC3 CV_MAKETYPE(CV_64F,3) - https://github.com/opencv/opencv/blob/05b15943d6a42c99e5f921b7dbaa8323f3c042c6/modules/core/include/opencv2/core/hal/interface.h#L120
# #define CV_MAKETYPE(depth,cn) (CV_MAT_DEPTH(depth) + (((cn)-1) << CV_CN_SHIFT))
# http://stackoverflow.com/questions/34355059/opencv-python-how-to-format-numpy-arrays-when-using-calibration-functions
#   The correct layout of objpoints is a list of numpy arrays with len(objpoints) = "number of pictures" and each entry beeing a numpy array.
#   Please have a look at the official help. OpenCV documentation talks about "vectors", which is equivalent of a list or numpy.array. In this instance a "vector of vectors" can be interpreted as a list of numpy.arrays.
# http://docs.opencv.org/3.1.0/db/d58/group__calib3d__fisheye.html#gad626a78de2b1dae7489e152a5a5a89e1
# http://docs.opencv.org/3.0-beta/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html

'''
if len(image_points) > 0:
    # カメラ内部パラメータを計算
    print('calibrateCamera() start')
    rms, K, d, r, t = cv2.calibrateCamera(object_points,image_points,(frame.shape[1],frame.shape[0]),None,None)
    print("RMS = ", rms)
    print("K = \n", K)
    print("d = ", d.ravel())
    np.savetxt("K.csv", K, delimiter =',',fmt="%0.14f") #カメラ行列の保存
    np.savetxt("d.csv", d, delimiter =',',fmt="%0.14f") #歪み係数の保存

    camera_mat = K
    dist_coef = d

    # 再投影誤差による評価
    mean_error = 0
    for i in range(len(object_points)):
        image_points2, _ = cv2.projectPoints(object_points[i], r[i], t[i], camera_mat, dist_coef)
        error = cv2.norm(image_points[i], image_points2, cv2.NORM_L2) / len(image_points2)
        mean_error += error
    print("total error: ", mean_error/len(object_points)) # 0に近い値が望ましい(魚眼レンズの評価には不適？)
else:
    print("findChessboardCorners() not be successful once")
'''

# 歪み補正画像表示
if K != []:
    undistort_image = cv2.fisheye.undistortImage(frame, K, D)

    # 縮小
    ratio = 1/4
    size = (int(frame.shape[1]*ratio), int(frame.shape[0]*ratio))
    
    cv2.imshow('original', cv2.resize(frame, size))
    cv2.imshow('undistort', cv2.resize(undistort_image, size))


cv2.waitKey(0)
cv2.destroyAllWindows()