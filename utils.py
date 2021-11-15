# -*- coding: utf-8 -*-
# @Time    : 2021/11/16 0:04
# @Author  : Marshall
# @FileName: utils.py
import cv2

def threshold_Img(path):
    img = cv2.imread(path, 0)  # 直接读为灰度图像
    # Otsu 滤波
    ret2, th2 = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return th2

if __name__ == '__main__':
    imgP = "./data/logo.jpg"
    re = threshold_Img(imgP)
    # 显示图片
    cv2.imshow('girl', re)
    cv2.waitKey(0)
    cv2.destroyAllWindows()