# https://github.com/***/***-glucose/blob/master/sphere/kanso_shiitake.py
'''
高速化を試みる

'''
import numpy as np
import cv2
from math import pi, sin, cos, ceil
import time


def KBT_func(image: np.ndarray)-> np.ndarray:
    w, h, _ = image.shape
    if w != h:
        v = max((w - h) / 2, 0)
        h = max((h - w) / 2, 0)
        padding = (
            (int(round(h)), int(ceil(h))),
            (int(round(v)), int(ceil(v))),
            (0, 0)
        )
        #print(padding)
        image = np.pad(image, padding, mode='constant')
    #print(image.shape)
    r2, _, p = image.shape
    w = int(pi * r2 / 10)
    h = int(r2 / 2)
    ANGLE = 36

    # おうぎ形で画像を切り抜くための座標群
    coordinates = []
    for r in range(h):
        length = int(2 * pi * r * ANGLE / 360)
        if length == 0:
            continue
        angle = 0
        row = []
        while angle < ANGLE:
            angle += ANGLE / length
            row.append(
                (
                    round(cos((252.0 + angle) / 360 * pi * 2) * r) + int(w / 2),
                    -round(sin((252.0 + angle) / 360 * pi * 2) * r)
                )
            )
        coordinates.append(row)

    # 幅の中央値を取得する
    width = int(np.median([len(m) for m in coordinates]))

    # 切り抜く、長方形に補正しつなげる、画像を回転する。のループ
    output = None
    for angle in range(0, 360, ANGLE):
        M = cv2.getRotationMatrix2D((r2 / 2, r2 / 2), angle, 1)
        rotate = cv2.warpAffine(image, M, (r2, r2))
        crop = rotate[h:, int(r2 / 2 - w / 2): int(r2 / 2 - w / 2 + w)]
        piece = []
        for m in coordinates:
            row = np.array([crop[y, x] for x, y in m])
            row = cv2.resize(row, (3, width))
            piece.append(row)
        piece = np.array(piece)
        if output is None:
            output = piece
        else:
            output = cv2.hconcat([piece, output])

    return output

def resize(ratio: float, image: np.ndarray)-> np.ndarray:
    size = (int(image.shape[1]*ratio), int(image.shape[0]*ratio))
    return cv2.resize(image, size)


if __name__ == '__main__':
    print("try loading")
    image_file = "./2016-10-18-123938.jpg"
    fisheye_img = cv2.imread(image_file)
    print(fisheye_img.shape)
    fisheye_img = resize(1/10, fisheye_img)
    print(fisheye_img.shape)
    print("loaded")
    i = 0
    prev = time.time()
    while(True):
        panorama_img = KBT_func(fisheye_img)
        cv2.imshow('fisheye_img', fisheye_img)
        cv2.imshow('panorama_img', panorama_img)
        
        i += 1
        if i%10 == 0:
            curr = time.time()
            diff = curr - prev
            print(i/diff, "fps")
            prev = curr
            i=0
            # 9fps!
        if cv2.waitKey(30) & 0xFF == ord('q'): break
        print("ok")

    print("fin")
    cap.release()
    cv2.destroyAllWindows()
    
