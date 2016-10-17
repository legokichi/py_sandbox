# original: http://qiita.com/Kazuhito/items/cccdeb55d101dcc69203
# reference1: http://qiita.com/Kazuhito/items/cccdeb55d101dcc69203
# reference2: http://kazuhito00.hatenablog.com/entry/2016/09/22/211109

import numpy as np
import cv2

chessboards = [
    "./Webcam2/2016-10-17-161547.jpg",
    "./Webcam2/2016-10-17-161553.jpg",
    "./Webcam2/2016-10-17-161557.jpg",
    "./Webcam2/2016-10-17-161601.jpg",
    "./Webcam2/2016-10-17-161609.jpg"
]

# 縮小


square_side_length = 24.0 # チェスボード内の正方形の1辺のサイズ(mm)
grid_intersection_size = (10, 7) # チェスボード内の格子数


# これ何？

pattern_points = np.zeros( (np.prod(grid_intersection_size), 3), np.float32 )
pattern_points[:,:2] = np.indices(grid_intersection_size).T.reshape(-1, 2)
'''
pattern_points = np.array([
    [0, 0, 0], [0, 1, 0], [0, 2, 0], [0, 3, 0], [0, 4, 0], [0, 5, 0], [0, 6, 0], [0, 7, 0], [0, 8, 0], [0, 9, 0],
    [1, 0, 0], [1, 1, 0], [1, 2, 0], [1, 3, 0], [1, 4, 0], [1, 5, 0], [1, 6, 0], [1, 7, 0], [1, 8, 0], [1, 9, 0],
    [2, 0, 0], [2, 1, 0], [2, 2, 0], [2, 3, 0], [2, 4, 0], [2, 5, 0], [2, 6, 0], [2, 7, 0], [2, 8, 0], [2, 9, 0],
    [3, 0, 0], [3, 1, 0], [3, 2, 0], [3, 3, 0], [3, 4, 0], [3, 5, 0], [3, 6, 0], [3, 7, 0], [3, 8, 0], [3, 9, 0],
    [4, 0, 0], [4, 1, 0], [4, 2, 0], [4, 3, 0], [4, 4, 0], [4, 5, 0], [4, 6, 0], [4, 7, 0], [4, 8, 0], [4, 9, 0],
    [5, 0, 0], [5, 1, 0], [5, 2, 0], [5, 3, 0], [5, 4, 0], [5, 5, 0], [5, 6, 0], [5, 7, 0], [5, 8, 0], [5, 9, 0],
    [6, 0, 0], [6, 1, 0], [6, 2, 0], [6, 3, 0], [6, 4, 0], [6, 5, 0], [6, 6, 0], [6, 7, 0], [6, 8, 0], [6, 9, 0]
], dtype=np.float32)
'''
print(pattern_points, pattern_points.shape, pattern_points.dtype)
pattern_points *= square_side_length



object_points = [] # type: List[<class 'numpy.ndarray'> (70, 3)]
image_points  = [] # type: List[<class 'numpy.ndarray'> (70, 1, 2)]

for i,chessboard in enumerate(chessboards):

    # 読み込み
    frame = cv2.imread(chessboard, cv2.IMREAD_GRAYSCALE) # type: np.ndarray (w, h, grayscale)

    # コーナー検出
    # corners # type: CvPoint2D32f[] = Nx[w,h]
    found, corners = cv2.findChessboardCorners(frame, grid_intersection_size) # type: bool, <class 'numpy.ndarray'> (70, 1, 2) 
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001) # termination criteria
    cv2.cornerSubPix(frame, corners, grid_intersection_size, (-1, -1), criteria)
    # 現在のOpenCVではfindChessboardCorners()内で、cornerSubPix()相当の処理が実施されている？要確認
    
    # 描画
    ratio = 1/4
    size = (int(frame.shape[1]*ratio), int(frame.shape[0]*ratio))
    cv2.drawChessboardCorners(frame, grid_intersection_size, corners, found)
    cv2.imshow(chessboard, cv2.resize(frame, size))



    pattern_points = pattern_points.reshape(1, 70, 3)
    corners = corners.reshape(1, 70, 2)
    print("object_points<pattern_points>:", type(pattern_points),  len(pattern_points),  pattern_points.shape, pattern_points.dtype)
    print("image_points <corners>       :", type(corners),  len(corners),  corners.shape, corners.dtype)

    # チェスボードコーナー検出情報を追加
    object_points.append(pattern_points)
    image_points.append(corners)

print(frame.shape)
# チェスボード撮影を終了し、カメラ内部パラメータを求めます
retval, K, D, rvecs, tvecs = cv2.fisheye.calibrate(object_points, image_points, image_size=(frame.shape[1],frame.shape[0]), K=None, D=None)#, flags=cv2.fisheye.CALIB_USE_INTRINSIC_GUESS)
'''
object_points,
    # M枚の画像上におけるK個の点それぞれの物理的な座標を格納したNx3の行列(N=KxM)
    # これを具体的に与えるのは困難なのでチェスボードのコーナー座標をカメラと正対した同一平面上にあるとして
    # (0,0),(0,1),(0,2),...,(1,0),(2,0),...,(1,1),...m(S_{width}-1,S_{height}-1)
    # object_pointsにチェスボードのコーナー座標（三次元）を各画像ごとに入れる。
    # 通常は世界座標系の原点をチェスボード上のどこかに固定してZ軸をボードの法線方向にするため、どの画像にも同一のコーナー座標群が入る。
image_points,
    # 上に対応するピクセル上の座標
    # findChessboardCorners()で検出した画像上のコーナー座標（サブピクセル精度）を入れる。
    # コーナーの並び順はobject_pointsと対応している必要
image_size=(frame.shape[1],frame.shape[0]),
    # 画像サイズ
    # チェスボードを撮影した画像のサイズ
K=None,
    # カメラ行列
    # [[f_x,   0, c_x],
    #  [ 0, f_y, c_y,],
    #  [ 0,   0,   1 ]]
D=None
    # レンズ歪み係数
    # [k_1, k_2, k_3, k_4]
retval
    # bool
rvecs
    # カメラ座標におけるチェスボードの回転軸と大きさで反時計回りの回転量を表すベクトル
tvecs
    # カメラ座標における平行移動量ベクトル 
'''
# # Error
# https://github.com/opencv/opencv/blob/master/modules/calib3d/src/fisheye.cpp#L720
# OpenCV Error: Assertion failed
#   (objectPoints.type() == CV_32FC3 || objectPoints.type() == CV_64FC3) in calibrate,
#   file /tmp/opencv3-20161008-27623-6s9pji/opencv-3.1.0/modules/calib3d/src/fisheye.cpp, line 695

# ## aside
# http://d.hatena.ne.jp/arche_t/20090120/1232445728
# IplImage     | CvMat    | 対応する他の構造体| チャンネルのバイト数 | チャンネル数 | 一要素のバイト数 | 符号 | 種類
# IPL_DEPTH_32F| CV_32FC3 | CvPoint3D32f   | 4                 | 3          | 12            | ?   | 浮動小数点数

# ## workaround
# https://github.com/opencv/opencv/issues/5534
# Transposing both object and image points to shapes
# (1, <num points in set> , 3) and (1, <num points in set> , 2),
# respectively, seems like a workaround for the above issue.

# ## c++ doc
# http://docs.opencv.org/3.1.0/db/d58/group__calib3d__fisheye.html#gad626a78de2b1dae7489e152a5a5a89e1

# ## cam model doc
# http://docs.opencv.org/3.0-beta/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html


# 歪み補正画像表示
'''
K=np.array([
    [100,0,frame.shape[0]/2],
    [0,100,frame.shape[1]/2],
    [0,0,1],
    ], dtype=np.float32)
D=np.array([[0],[0],[0],[0]], dtype=np.float32)
'''

print("K:", K, type(K))
print("D:", D, type(D))
print("rv:", rvecs, type(rvecs))
print("tv:", tvecs, type(tvecs))


if K != []:
    frame = cv2.imread(chessboards[0], cv2.IMREAD_GRAYSCALE)
    undistort_image = cv2.fisheye.undistortImage(frame, K, D)

    ratio = 1/4
    size = (int(frame.shape[1]*ratio), int(frame.shape[0]*ratio))


    cv2.imshow('original', cv2.resize(frame, size))
    cv2.imshow('undistort', cv2.resize(undistort_image, size))


cv2.waitKey(0)
cv2.destroyAllWindows()
