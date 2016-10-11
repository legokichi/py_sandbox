import numpy as np
import cv2

# パターン
chessboard = "../Webcam/2016-10-11-154140.jpg"
patternSize = (10, 7)

# 読み込み
img = cv2.imread(chessboard, cv2.IMREAD_GRAYSCALE)

# コーナー検出
ret, corners = cv2.findChessboardCorners(img, patternSize)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001) # termination criteria
cv2.cornerSubPix(img, corners, patternSize, (-1,-1), criteria)

# 魚眼レンズのカメラ行列を推定

'''
http://docs.opencv.org/3.1.0/db/d58/group__calib3d__fisheye.html#gad626a78de2b1dae7489e152a5a5a89e1
cv2.fisheye.calibrate
auto fisheye::calibrate(
  InputArrayOfArrays objectPoints,
    // vector of vectors of calibration pattern points in the calibration pattern coordinate space.
    // object_pointsにチェスボードのコーナー座標（三次元）を各画像ごとに入れる。
    // 通常は世界座標系の原点をチェスボード上のどこかに固定してZ軸をボードの法線方向にするため、どの画像にも同一のコーナー座標群が入る。
  InputArrayOfArrays imagePoints,
    // vector of vectors of the projections of calibration pattern points. imagePoints.size() and objectPoints.size() and imagePoints[i].size() must be equal to objectPoints[i].size() for each i.
    // findChessboardCorners()で検出した画像上のコーナー座標（サブピクセル精度）を入れる。
	  // コーナーの並び順はobject_pointsと対応している必要
  const Size& image_size,
    // Size of the image used only to initialize the intrinsic camera matrix.
    // チェスボードを撮影した画像のサイズ
  InputOutputArray K,              // Output 3x3 floating-point camera matrix ...
  InputOutputArray D,              // Output vector of distortion coefficients (k1,k2,k3,k4).
  OutputArrayOfArrays rvecs,       // rvecs	Output vector of rotation vectors (see Rodrigues ) estimated for each pattern view. That is, each k-th rotation vector together with the corresponding k-th translation vector (see the next output parameter description) brings the calibration pattern from the model coordinate space (in which object points are specified) to the world coordinate space, that is, a real position of the calibration pattern in the k-th pattern view (k=0.. M -1).
  OutputArrayOfArrays tvecs,       // Output vector of translation vectors estimated for each pattern view.
  int flags=0,                     // Different flags that may be zero or a combination of the following values:
  TermCriteria criteria=TermCriteria(TermCriteria::COUNT + TermCriteria::EPS, 100, DBL_EPSILON)
)-> double
'''

# 描画
cv2.drawChessboardCorners(img, patternSize, corners, ret)
ratio = 1/4
size = (int(img.shape[1]*ratio), int(img.shape[0]*ratio))
img = cv2.resize(img, size)

cv2.imshow('image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()