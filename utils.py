# -*- coding: utf-8 -*-
# @Time    : 2021/11/16 0:04
# @Author  : Marshall
# @FileName: utils.py
import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage import measure
import xlwt
import os
import seaborn as sns


class Configs():
    out_list = "ALL"
    is_full = False
    scale = None
    dis = 10
    out = True

def threshold_Img(path):
    img = cv2.imread(path, 0)  # 直接读为灰度图像
    # Otsu 滤波
    ret2, th2 = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return th2

def preprocessing(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    h, w = img.shape
    # 去掉底部信息 同时计算标尺像素长度
    img_half = abs(255 - img[:, :int(w / 2)])
    img_half_csum = np.sum(img_half, axis=-1)
    end_h = np.where(img_half_csum == 0)[0][0]
    img_full = img[:end_h, :]
    img_bottom = img[end_h:h, :]
    img_bottom_csum = np.sum(img_bottom, axis=-1)
    scale_row = np.argmin(img_bottom_csum)
    pixels = len(np.where(img_bottom[scale_row, :] == 0)[0])
    return img_full, pixels


def batch_export(root, out_dir="out", configs=None):
    if configs is None:
        configs = Configs()
    if os.path.isdir(root):
        file_dir = os.listdir(root)
    else:
        file_dir = [None]

    workbook = xlwt.Workbook()  # 新建一个工作簿
    sheet = workbook.add_sheet("检测值")  # 在工作簿中新建一个表格
    if configs.out_list == "ALL":
        sheet.write(0, 0, "文件名")
        sheet.write(0, 1, "平均直径")
        sheet.write(0, 2, "平均直径(像素)")
        sheet.write(0, 3, "直径cv")
        sheet.write(0, 4, "孔隙率")
        sheet.write(0, 5, "孔隙cv")
    else:
        pass
    i = 1
    for file_path in file_dir:
        if file_path is None:
            img_path = root
        else:
            img_path = os.path.join(root, file_path)
        if not configs.is_full:
            img, scale = preprocessing(img_path)
        else:
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            scale = configs.scale
        reults_dic, reults_fig = sem_analysis(img)
        diameter, diameter_cv, porosity, porosity_cv = reults_dic["diameter"],reults_dic["diameter_cv"],reults_dic["porosity"],reults_dic["porosity_cv"]
        # cv2.imwrite(os.path.join(out_dir, file_path), img)
        (filepath, tempfilename) = os.path.split(img_path)
        file_name = os.path.splitext(tempfilename)[0]
        out_dir = os.path.join(filepath,file_name + "_" + out_dir)
        if not os.path.exists(out_dir):
            os.mkdir(out_dir)
        if configs.out:
            cv2.imwrite(os.path.join(out_dir, file_name+"_full.png"), img)
            cv2.imwrite(os.path.join(out_dir, file_name + "_th2.png"), reults_fig["th2"])
            cv2.imwrite(os.path.join(out_dir, file_name + "_dilated.png"), reults_fig["dilated"])
            reults_fig["D"].savefig(os.path.join(out_dir, file_name + "_D.png"),dpi=300)
        DPI = configs.dis / scale
        real_diameter = DPI * diameter
        sheet.write(i, 0, file_path)
        sheet.write(i, 1, real_diameter)
        sheet.write(i, 2, diameter)
        sheet.write(i, 3, diameter_cv)
        sheet.write(i, 4, porosity)
        sheet.write(i, 5, porosity_cv)
        i = i + 1
        print("###" * 10)
        print(file_path, "平均直径:", real_diameter)
        print(file_path, "直径cv:", diameter_cv)
        print(file_path, "孔隙率:", porosity)
        print(file_path, "孔隙cv:", porosity_cv)
    workbook.save(os.path.join(out_dir, "results.xls"))  # 保存工作簿

def sem_analysis(img):
    # 高斯模糊去噪点
    image = cv2.GaussianBlur(img, ksize=(0, 0), sigmaX=0.5, borderType=cv2.BORDER_REPLICATE)
    # plt.imshow(image, "gray")
    # plt.title("GaussianBlur")
    # plt.show()
    # 大津法寻找合适阈值
    ret2, th2 = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # plt.imshow(th2, "gray")
    # plt.title("THRESH_OTSU")
    # plt.show()
    # 开闭运算去噪音
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    eroded = cv2.erode(th2, kernel)  # 腐蚀图像
    dilated = cv2.dilate(eroded, kernel)  # 膨胀图像
    # plt.imshow(dilated, "gray")
    # plt.title("MORPH")
    # plt.show()

    h, w = image.shape
    # 计算纱线直径
    D = []
    for i in range(h):
        L = image[i, :]
        L = L > np.mean(L)
        L = measure.label(L)
        area = []
        for l in range(max(L)):
            if l > 0:
                area.append(np.sum(L == l))
        D.append(np.mean(area))

    for i in range(w):
        L = image[:, i]
        L = L > np.mean(L)
        L = measure.label(L)
        area = []
        for l in range(max(L)):
            if l > 0:
                area.append(np.sum(L == l))
        D.append(np.mean(area))
    D = np.sort(D)
    D = D[int(len(D) * 0.01):int(len(D) * 0.99)]
    # 计算平均直径
    diameter = np.mean(D)
    # 计算直径均匀性
    diameter_cv = np.std(D) / np.mean(D)
    # 绘制直径分布曲线
    sns.set_style("darkgrid")
    fig = sns.distplot(D)
    dist_fig = fig.get_figure()
    # plt.title("Fiber Diameter Distribution")
    # plt.xlabel("Fiber Diameter (pixel)")
    # plt.ylabel("Relative frequency")
    # plt.show()

    # 计算孔隙率
    porosity = 1 - np.sum(dilated / 255) / (h * w)
    # 计算空隙大小均匀性
    areas = []
    contours, hierarchy = cv2.findContours(dilated.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        if cv2.contourArea(contour) < h * w * 0.5:  # 面积阈值
            areas.append(cv2.contourArea(contour))
    porosity_cv = np.std(areas) / np.mean(areas)
    reults_dic = {"diameter":diameter, "diameter_cv":diameter_cv, "porosity":porosity, "porosity_cv":porosity_cv}
    reults_fig = {"th2":th2, "dilated":dilated, "D":dist_fig}
    return reults_dic, reults_fig


if __name__ == '__main__':
    imgP = "./data/logo.jpg"
    re = threshold_Img(imgP)
    # 显示图片
    cv2.imshow('girl', re)
    cv2.waitKey(0)
    cv2.destroyAllWindows()