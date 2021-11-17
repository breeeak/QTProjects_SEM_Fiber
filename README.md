# 使用pyqt opencv 等实现 sem图像中 纤维直径等的分析
## 初始版本
使用传统方法进行分析

## 安装
```markdown
pip freeze >requirements.txt
pip install -r requirements.txt

# pyinstaller -F -w --icon=icon.ico --paths="D:\\3_Research\\2_Learning\\Codes\\Python\\QTProjects\\venv" mainWindow.py
pyinstaller -w --icon=icon.ico --paths="D:\\3_Research\\2_Learning\\Codes\\Python\\QTProjects\\venv\\site-packages\\cv2" mainWindow.py
```