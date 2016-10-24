import cv2
import numpy as np
import dlib


def histogram_equalize(img):
    # http://stackoverflow.com/questions/31998428/opencv-python-equalizehist-colored-image
    b, g, r = cv2.split(img)
    red = cv2.equalizeHist(r)
    green = cv2.equalizeHist(g)
    blue = cv2.equalizeHist(b)
    return cv2.merge((blue, green, red))

def resize_show(ratio:float, name:str, frame):
  size = (int(frame.shape[1]*ratio), int(frame.shape[0]*ratio))
  cv2.imshow(name, cv2.resize(frame, size))

def filter(img_src):
    # https://www.blog.umentu.work/python-opencv3%E3%81%A7bilateral%E3%82%AA%E3%83%9A%E3%83%AC%E3%83%BC%E3%82%BF%E3%82%92%E4%BD%BF%E3%81%A3%E3%81%A6%E5%B9%B3%E6%BB%91%E5%8C%96/
    # 平均化する画素の周囲の大きさを指定する。
    # 25の場合、個々の画素の地点の周囲25×25マスの平均をとる。
    # 数値が大きいほどぼやける。
    average_square_size = 100

    # 色空間に関する標準偏差
    sigma_color = 1

    # 距離空間に関する標準偏差
    sigma_metric = 1

    # Bilateralオペレータを使用して平滑化
    # http://docs.opencv.org/3.0-beta/modules/imgproc/doc/filtering.html#bilateralfilter
    img_bilateral = cv2.bilateralFilter(img_src, 
                             average_square_size,
                             sigma_color,
                             sigma_metric)
    return img_bilateral

def gamma_filter(img_src, gamma = 1.8):
    # http://peaceandhilightandpython.hatenablog.com/entry/2016/02/05/004445
    lookUpTable = np.zeros((256, 1), dtype = 'uint8')

    for i in range(256):
        lookUpTable[i][0] = 255 * pow(float(i) / 255, 1.0 / gamma)

    img_gamma = cv2.LUT(img_src, lookUpTable)
    return img_gamma

def detect(frame):
      detector = dlib.get_frontal_face_detector() # type: http://dlib.net/python/#dlib.fhog_object_detector
      predictor_path = "shape_predictor_68_face_landmarks.dat" # http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2 を解凍
      predictor = dlib.shape_predictor(predictor_path)
      image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # RGB変換 (opencv形式からskimage形式に変換)
      # detsが矩形, scoreはスコア、idxはサブ検出器の結果(0.0がメインで数が大きい程弱い)
      dets, scores, idx = detector.run(image, 1) # 1 は upsample_num_times
      if len(dets) > 0: # 顔画像ありと判断された場合
          print("face detected.")
          for i, rect in enumerate(dets):
              # 矩形描画
              cv2.rectangle(frame, (rect.left(), rect.top()), (rect.right(), rect.bottom()), (0, 0, 255), thickness=1)
              # 輪郭検出
              shape = predictor(image, rect) # type: full_object_detection # http://dlib.net/python/#dlib.full_object_detection
              pt2 = shape.part(shape.num_parts - 1)
              for j, pt in enumerate(shape.parts()): # type: int, dlib.point # http://dlib.net/python/#dlib.point
                  cv2.line(frame, (pt.x, pt.y), (pt2.x, pt2.y), (0, 255, 0), thickness=1)
                  pt2 = pt
      return frame
      #cv2.imwrite('panorama/{}.jpg'.format(time.time()), frame)
      # cv2.waitKey(0) # 1ms 待って imshow を描画

if __name__ == "__main__":
    frame = cv2.imread("./2001Z_01.panorama.jpg")
    #frame1 = gamma_filter(frame, 1)
    #frame2 = gamma_filter(frame, 2)
    #frame3 = gamma_filter(frame, 3)
    #merge_mertens = cv2.createMergeMertens()
    #frame4 = merge_mertens.process([frame,frame1,frame2,frame3])
    
    resize_show(1/2, "a", detect(frame))
    #resize_show(1/2, "b", detect(frame1))
    #resize_show(1/2, "c", detect(frame2))
    #resize_show(1/2, "d", detect(frame3))
    #resize_show(1/2, "e", detect(frame4))
    cv2.waitKey(0)