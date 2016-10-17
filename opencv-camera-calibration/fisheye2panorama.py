# http://www.atrialogic.com/dewarping.php
import numpy as np
import cv2

chessboard = "./Webcam/2016-10-03-205226.jpg"
chessboard = "./Webcam/2016-10-11-154109.jpg"
frame = cv2.imread(chessboard)

ratio = 1/4; size = (int(frame.shape[1]*ratio), int(frame.shape[0]*ratio))
cv2.imshow('source', cv2.resize(frame, size))
#frame = cv2.resize(frame, size)

Hs,Ws,_ = frame.shape # fisheye 画像短径
Cx,Cy = (Ws/2,Hs/2)     # fisheye 中心座標
R1,R2 = 0,Hs/2       # fisheye から ドーナッツ状に切り取る領域を決める半径二つ
Wd,Hd = int((R2+R1)*np.pi),int(R2-R1) # ドーナッツ状に切り取った領域を短径に変換した大きさ

print("size:", Ws,Hs)
print("centor:", Cx,Cy)
print("radius:", R1,R2)
print("dest:", Wd, Hd)

dest = np.zeros((Hd, Wd, 3), dtype=np.uint8)
print("dest.shape:", dest.shape)

for xD in range(Wd):
  for yD in range(Hd):
    r = (float(yD)/float(Hd))*(R2-R1)+R1
    theta = (float(xD)/float(Wd))*2.0*np.pi
    xS = int(Cx+r*np.sin(theta))
    yS = int(Cy+r*np.cos(theta))
    #print("plr:", r, theta)
    #print("src:", xS, yS)
    #print("dst:", xD, yD)
    dest[yD][xD] = frame[yS][xS]
    #print(frame[yS][xS], dest[yD][xD])
    


ratio = 1/4; size = (int(dest.shape[1]*ratio), int(dest.shape[0]*ratio))
cv2.imshow('destination', cv2.resize(dest, size))

cv2.waitKey(0)
cv2.destroyAllWindows()