# -*- coding: utf-8 -*-
# @Time    : 2021/11/15 16:10
# @Author  : Marshall
# @FileName: mainWindow.py
from PyQt5.QtGui import QPixmap,QImage

from sem_main import *
from utils import *

import sys
import os
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileSystemModel, QFileDialog, QMdiSubWindow, QTextEdit, QLabel, \
    QMessageBox


class SEMMainWindow(QMainWindow,Ui_SEMWindow):
    def __init__(self,parent=None):
        super(SEMMainWindow,self).__init__(parent)
        self.setupUi(self)
        # 当前的路径
        self.project_dir = os.path.dirname(os.path.abspath(__file__))
        self.file_name = None
        # 先初始化路径
        self.file_model = QFileSystemModel()
        # self.file_model.setRootPath(self.project_dir)
        self.treeView.setModel(self.file_model)
        # self.treeWidget.setRootIndex(self.fileModel.index(project_dir))
        # 打开的subwinwow数目
        self.subWindowNum = 0
        # 绑定一些事件
        self.actionExit.triggered.connect(self.close)
        self.actionOpen.triggered.connect(self.openFile0rDirectory)
        self.actionProcess.triggered.connect(self.processImg)
        self.setting_window.setVisible(False)
        self.actionSetting.triggered.connect(self.setting_window.show)
        self.pushButton_Save.clicked.connect(self.setting_save)
        self.pushButton_Reset.clicked.connect(self.setting_reset)

    def setting_save(self):
        self.msg_success()
        self.setting_window.close()
    def setting_reset(self):
        self.msg_success()
        self.setting_window.close()

    def msg_success(self):
        # 使用infomation信息框
        QMessageBox.information(self, "提示信息", "保存成功！")

    def setPath(self,path):
        """
        设置当前的根节点
        :param path:
        :return:
        """
        self.file_model.setRootPath(path)
        self.treeView.setRootIndex(self.file_model.index(path))


    def openFile0rDirectory(self):
        fileName_choose, filetype = QFileDialog.getOpenFileName(self, "选取文件", self.project_dir,"Images (*.png *.xpm *.jpg *.tif)")  # 起始路径
        self.project_dir = os.path.dirname(fileName_choose)
        self.setPath(self.project_dir)
        self.treeView.setCurrentIndex(self.file_model.index(fileName_choose))
        self.file_name = fileName_choose
        self.loadImage(fileName_choose)

    def loadImage(self, img_path,title="Original Image"):
        sub = QMdiSubWindow()
        pic = QPixmap(img_path)
        lb = QLabel()
        lb.setPixmap(pic)
        lb.setScaledContents(True)
        sub.setWidget(lb)
        sub.setMaximumSize(400,400)
        sub.setWindowTitle(title)
        self.actionClear_Sub_Windows.triggered.connect(sub.close)
        sub.setObjectName("sub_" + str(self.subWindowNum))
        self.mdiArea.addSubWindow(sub)
        sub.show()




    def processImg(self):
        result_img = threshold_Img(self.file_name)
        # image2 = QImage(result_img, result_img.shape[1], result_img.shape[0], result_img.shape[1] * 3,
        #                 QImage.Format_RGB888)  # 参数依次为：图像、宽、高、每一行的字节数、图像格式彩色图像一般为Format_RGB888
        image_processed = QImage(result_img, result_img.shape[1], result_img.shape[0],result_img.shape[1],QImage.Format_Indexed8) ##这是灰度图像
        batch_export(self.file_name)
        file_name = os.path.splitext(self.file_name)[0]
        (filepath, tempfilename) = os.path.split(file_name)
        self.loadImage(os.path.join(file_name+"_out", tempfilename+"_full.png"), "Processed Image")
        self.loadImage(os.path.join(file_name+"_out", tempfilename+"_th2.png"), "Threshed Image")
        self.loadImage(os.path.join(file_name+"_out", tempfilename+"_dilated.png"), "Dilated Image")
        self.loadImage(os.path.join(file_name+"_out", tempfilename+ "_D.png"), "Distribution")
        # self.loadImage(image_processed, "Distribution")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    MainWindow = SEMMainWindow()
    MainWindow.show()
    sys.exit(app.exec_())

