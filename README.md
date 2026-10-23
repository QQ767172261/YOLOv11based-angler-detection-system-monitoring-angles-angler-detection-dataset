监控视角下钓鱼人员检测数据集，基于YOLOv11的垂钓者检测系统![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/7737485ee1ee4fce96335b87b183f1e8.png)
328张，yolo，voc，coco三种标注方式
图像尺寸:640*640
类别数量:1类![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/767d2033d4b148a79f051396d49b8346.png)

训练集图像数量:230; 验证集图像数量:65； 测试集图像数量:33
类别名称: 每一类图像数 ，每一类标注数
fishing: 322,400![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/f78b3f9637694fa4bb14cc78e62932ef.png)

image num: 328

模型代码：模型训练使用yolov11n训练，50个epoch训练结果，map如描述图所示。![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/84569dace855492a914a421611741c2e.png)

qt界面：运行界面采用pyqt编写，本项目已经训练好模型，配置好环境后可直接使用，运行效果见描述图像![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/9b19735ddfaa46aa9ebbea11aab1f28f.png)
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/af159ca61bd34e40b4cab05a38da5f6d.png)
# 基于YOLOv11的垂钓者检测系统
## 1、环境安装
```bash
pip install ultralytics opencv-python pyqt5 numpy
```

## 2、数据集配置 fishing.yaml
```yaml
path: ./fishing_dataset
train: images/train
val: images/val
test: images/test

names:
  0: fishing
```

## 3、模型训练脚本 train.py
```python
from ultralytics import YOLO

if __name__ == '__main__':
    # 加载YOLOv11检测模型
    model = YOLO("yolo11n.pt")
    results = model.train(
        data="fishing.yaml",
        epochs=80,
        imgsz=640,
        batch=8,
        device=0,
        workers=2,
        amp=False,
        project="runs/train",
        name="fishing_det"
    )
    print("训练完成！")
    metrics = model.val()
    print(f"mAP@0.5:{metrics.box.map50}")
```

## 4、推理脚本 predict.py
```python
from ultralytics import YOLO
import cv2

model = YOLO("runs/train/fishing_det/weights/best.pt")

def predict_image(img_path):
    res = model(img_path, conf=0.3)
    for r in res:
        img = r.plot()
        cv2.imwrite("result.jpg", img)
    return img

predict_image("test.jpg")
# model.predict("test.mp4", save=True, conf=0.3)
```

## 5、PyQt5可视化界面代码 main_ui.py
```python
import sys
import cv2
import pandas as pd
from PyQt5.QtWidgets import (QApplication, QMainWindow, QPushButton, QLabel, QFileDialog, QWidget,
                             QHBoxLayout, QVBoxLayout, QTableWidget, QTableWidgetItem, QComboBox)
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from ultralytics import YOLO

model = YOLO("runs/train/fishing_det/weights/best.pt")

class VideoThread(QThread):
    change_pixmap = pyqtSignal(QImage)
    def run(self):
        cap = cv2.VideoCapture(0)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            res = model(frame, conf=0.3)
            frame = res[0].plot()
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h,w,ch = rgb.shape
            qimg = QImage(rgb.data, w, h, ch*w, QImage.Format_RGB888)
            self.change_pixmap.emit(qimg)
        cap.release()

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("基于YOLOv11的垂钓者检测系统")
        self.setGeometry(100,100,1200,850)
        self.thread = None
        self.result_data = []

        w = QWidget()
        self.setCentralWidget(w)
        main_layout = QHBoxLayout(w)

        # 左侧图片区域
        left_layout = QVBoxLayout()
        self.img_label = QLabel("图像展示区域")
        self.img_label.setFixedSize(700,500)
        left_layout.addWidget(self.img_label)

        # 结果表格
        self.table = QTableWidget()
        self.table.setColumnCount(5)
        self.table.setHorizontalHeaderLabels(["序号","文件路径","类别","置信度","坐标位置"])
        left_layout.addWidget(self.table)

        # 右侧控制面板
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.addWidget(QLabel("文件导入"))
        self.btn_img = QPushButton("选择图片")
        self.btn_video = QPushButton("选择视频")
        self.btn_cam = QPushButton("开启摄像头")
        self.btn_img.clicked.connect(self.load_img)
        self.btn_video.clicked.connect(self.load_video)
        self.btn_cam.clicked.connect(self.open_cam)
        right_layout.addWidget(self.btn_img)
        right_layout.addWidget(self.btn_video)
        right_layout.addWidget(self.btn_cam)

        right_layout.addWidget(QLabel("检测结果"))
        self.label_time = QLabel("用时：-- s  目标数目：--")
        self.label_conf = QLabel("置信度 -- %")
        self.label_pos = QLabel("xmin: -- ymin: --\nxmax: -- ymax: --")
        self.combo_target = QComboBox()
        self.combo_target.addItem("全部")
        right_layout.addWidget(self.label_time)
        right_layout.addWidget(self.combo_target)
        right_layout.addWidget(self.label_conf)
        right_layout.addWidget(self.label_pos)

        self.btn_save = QPushButton("保存")
        self.btn_exit = QPushButton("退出")
        self.btn_save.clicked.connect(self.save_result)
        self.btn_exit.clicked.connect(self.close)
        right_layout.addWidget(self.btn_save)
        right_layout.addWidget(self.btn_exit)

        main_layout.addLayout(left_layout)
        main_layout.addWidget(right_widget)

    def load_img(self):
        file,_ = QFileDialog.getOpenFileName(self,"打开图片","","*.jpg;*.png;*.bmp")
        if not file:
            return
        src = cv2.imread(file)
        res = model(file,conf=0.3)[0]
        img_res = res.plot()
        rgb_res = cv2.cvtColor(img_res,cv2.COLOR_BGR2RGB)
        h,w,ch = rgb_res.shape
        q_res = QImage(rgb_res.data,w,h,ch*w,QImage.Format_RGB888)
        self.img_label.setPixmap(QPixmap.fromImage(q_res))

        # 更新右侧信息+表格
        boxes = res.boxes
        self.result_data.clear()
        self.table.setRowCount(0)
        for idx, box in enumerate(boxes):
            conf = float(box.conf)
            cls = res.names[int(box.cls)]
            xyxy = box.xyxy.tolist()[0]
            x1,y1,x2,y2 = map(int,xyxy)
            self.result_data.append([idx+1, file, cls, f"{conf:.2f}", f"[{x1},{y1},{x2},{y2}]"])
            self.table.insertRow(idx)
            self.table.setItem(idx,0,QTableWidgetItem(str(idx+1)))
            self.table.setItem(idx,1,QTableWidgetItem(file))
            self.table.setItem(idx,2,QTableWidgetItem(cls))
            self.table.setItem(idx,3,QTableWidgetItem(f"{conf:.4f}"))
            self.table.setItem(idx,4,QTableWidgetItem(f"[{x1},{y1},{x2},{y2}]"))
        if len(boxes)>0:
            self.label_time.setText(f"用时：{res.speed['inference']/1000:.3f} s  目标数目：{len(boxes)}")
            box0 = boxes[0]
            self.label_conf.setText(f"置信度{float(box0.conf)*100:.2f} %")
            x1,y1,x2,y2 = map(int,box0.xyxy.tolist()[0])
            self.label_pos.setText(f"xmin: {x1}   ymin: {y1}\nxmax: {x2}   ymax: {y2}")

    def load_video(self):
        path,_ = QFileDialog.getOpenFileName(self,"打开视频","","*.mp4;*.avi")
        if path:
            model.predict(path,save=True,conf=0.3)

    def open_cam(self):
        self.thread = VideoThread()
        self.thread.change_pixmap.connect(lambda img:self.img_label.setPixmap(QPixmap.fromImage(img)))
        self.thread.start()

    def save_result(self):
        df = pd.DataFrame(self.result_data,columns=["序号","文件路径","类别","置信度","坐标位置"])
        df.to_csv("det_result.csv",index=False,encoding="utf-8-sig")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())
```

运行：`python main_ui.py`
功能：图片/视频/摄像头实时垂钓者（fishing）目标检测，输出检测框、置信度、坐标，结果表格展示并支持导出csv，和截图界面保持一致。

