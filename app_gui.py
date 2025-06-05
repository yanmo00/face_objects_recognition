# import sys
# import cv2
# import torch
# import numpy as np
# from datetime import datetime
# from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QVBoxLayout
# from PyQt5.QtCore import Qt, QThread, pyqtSignal, QRect, QSize
# from PyQt5.QtGui import QImage, QPixmap
#
# from ultralytics import YOLO
# from models.mtcnn import MTCNN
# from models.face_alignment import Alignment
# from models.inception_resnet_v1 import InceptionResnetV1
# from get_database_faces_feature import get_database_faces_feature
# from PIL import Image, ImageDraw, ImageFont
# from shapely.geometry import box as shapely_box
#
#
# class VideoThread(QThread):
#     frame_data = pyqtSignal(np.ndarray)
#
#     def __init__(self):
#         super().__init__()
#         self.running = True
#
#         # 初始化模型
#         self.mtcnn = MTCNN(keep_all=True, post_process=False, device='cpu')
#         self.resnet = InceptionResnetV1(pretrained=True).eval()
#         self.faces_feature_database = get_database_faces_feature('./data/database_images')
#         self.faces_name_database = [i[0] for i in self.faces_feature_database]
#         self.features_database = [i[1] for i in self.faces_feature_database]
#
#         self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
#         try:
#             self.yolo_model = YOLO('./yolo/weights/best20250530.pt').to(self.device)
#             print("YOLO模型加载成功，检测类别:", self.yolo_model.names)
#         except Exception as e:
#             print(f"YOLO加载失败: {str(e)}")
#             self.yolo_model = None
#
#         self.crossing_line_x = 320
#         self.crossed_faces = set()
#         self.person_objects = {}
#
#         self.cap = cv2.VideoCapture(1)
#         self.frame_width = 1280
#         self.frame_height = 720
#         self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
#         self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)
#
#     def run(self):
#         while self.running:
#             ret, frame = self.cap.read()
#             if not ret:
#                 continue
#
#             yolo_results = []
#             if self.yolo_model:
#                 try:
#                     yolo_detections = self.yolo_model.predict(source=frame, conf=0.3, imgsz=640, device=self.device, verbose=False)
#                     if yolo_detections:
#                         yolo_results = yolo_detections[0].boxes.data.cpu().numpy()
#                 except Exception as e:
#                     print(f"YOLO检测出错: {str(e)}")
#
#             image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#             faces, prob, bboxs, points = self.mtcnn(image, return_prob=True)
#
#             img_draw = Image.fromarray(image)
#             draw = ImageDraw.Draw(img_draw)
#
#             # 画越界线
#             draw.line([(self.crossing_line_x, 0), (self.crossing_line_x, self.frame_height)], fill=(255, 0, 0), width=5)
#
#             if self.yolo_model and len(yolo_results) > 0:
#                 for det in yolo_results:
#                     x1, y1, x2, y2, conf, cls = det
#                     # label = f"{self.yolo_model.names[int(cls)]} {conf:.2f}"
#                     label = f"{self.yolo_model.names[int(cls)]} "
#                     draw.rectangle([x1, y1, x2, y2], outline="red", width=5)
#                     ft = ImageFont.truetype('bgtx.ttf', size=30)
#                     draw.text((x1, max(y1 - 30, 0)), text=label, font=ft, fill='red')
#
#             if faces is not None:
#                 faces_after_aligned = []
#                 for face, point in zip(faces, points):
#                     face = face.permute(1, 2, 0).numpy().astype(np.uint8)
#                     face_aligned, _ = Alignment(face, point)
#                     face_aligned = torch.tensor(face_aligned, dtype=torch.float32).permute(2, 0, 1)
#                     face_aligned = (face_aligned - 127.5) / 128
#                     faces_after_aligned.append(face_aligned)
#
#                 if len(faces_after_aligned) > 0:
#                     faces_stack = torch.stack(faces_after_aligned)
#                     faces_feature = self.resnet(faces_stack)
#                     dist = np.array([[(e1 - e2).norm().item() for e1 in self.features_database] for e2 in faces_feature])
#                     dist_min = dist.min(axis=1)
#                     idx_min = dist.argmin(axis=1)
#                     is_who = [self.faces_name_database[idx] if num < 0.7 else 'Unknown' for idx, num in zip(idx_min, dist_min)]
#
#                     for box, point, text in zip(bboxs, points, is_who):
#                         rate = max(1, (box[2] - box[0]) // 100)
#                         draw.rectangle(box.tolist(), width=2 * int(rate))
#                         ft = ImageFont.truetype('bgtx.ttf', size=20 * int(rate))
#                         draw.text(box[:2], text=text, font=ft, fill='blue')
#
#                         face_center_x = (box[0] + box[2]) // 2
#                         current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
#
#                         person_items = []
#                         face_box = shapely_box(box[0], box[1], box[2], box[3])
#
#                         if self.yolo_model and len(yolo_results) > 0:
#                             for det in yolo_results:
#                                 x1, y1, x2, y2, conf, cls = det
#                                 item_box = shapely_box(x1, y1, x2, y2)
#                                 if face_box.intersects(item_box):
#                                     person_items.append(self.yolo_model.names[int(cls)])
#
#                         if face_center_x > self.crossing_line_x:
#                             if text != 'Unknown' and text not in self.crossed_faces:
#                                 self.crossed_faces.add(text)
#                                 self.person_objects[text] = person_items
#                                 print(f"{text} 在 {current_time} 拿了 {person_items} 入库")
#                         elif face_center_x < self.crossing_line_x:
#                             if text in self.crossed_faces:
#                                 self.crossed_faces.remove(text)
#                                 items = self.person_objects.get(text, [])
#                                 print(f"{text} 在 {current_time} 拿了 {items} 出库")
#                                 self.person_objects.pop(text, None)
#
#             result_frame = cv2.cvtColor(np.array(img_draw), cv2.COLOR_RGB2BGR)
#             self.frame_data.emit(result_frame)
#
#         self.cap.release()
#
#
# class MainWindow(QWidget):
#     def __init__(self):
#         super().__init__()
#         self.setWindowTitle("Security System")
#         self.resize(1280, 720)
#         self.setFixedSize(1280, 720)  # 固定大小
#
#         self.label = QLabel(self)
#         self.label.setAlignment(Qt.AlignCenter)
#         self.label.resize(1280, 720)
#
#         # 居中显示窗口
#         qr = self.frameGeometry()
#         cp = QApplication.primaryScreen().availableGeometry().center()
#         qr.moveCenter(cp)
#         self.move(qr.topLeft())
#
#         self.thread = VideoThread()
#         self.thread.frame_data.connect(self.update_image)
#         self.thread.start()
#
#     def update_image(self, cv_img):
#         """将OpenCV图像转换成Qt图像显示"""
#         rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
#         h, w, ch = rgb_image.shape
#         bytes_per_line = ch * w
#         qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
#         pixmap = QPixmap.fromImage(qt_image)
#         self.label.setPixmap(pixmap.scaled(self.label.size(), Qt.KeepAspectRatio))
#
#
#     def closeEvent(self, event):
#         self.thread.running = False
#         self.thread.wait()
#         event.accept()
#
#
# if __name__ == '__main__':
#     app = QApplication(sys.argv)
#     w = MainWindow()
#     w.show()
#     sys.exit(app.exec_())
import sys
import cv2
import torch
import numpy as np
from datetime import datetime
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QVBoxLayout
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QRect, QSize
from PyQt5.QtGui import QImage, QPixmap

from ultralytics import YOLO
from models.mtcnn import MTCNN
from models.face_alignment import Alignment
from models.inception_resnet_v1 import InceptionResnetV1
from get_database_faces_feature import get_database_faces_feature
from PIL import Image, ImageDraw, ImageFont
from shapely.geometry import box as shapely_box


class VideoThread(QThread):
    frame_data = pyqtSignal(np.ndarray)

    def __init__(self):
        super().__init__()
        self.running = True

        # 初始化模型
        self.mtcnn = MTCNN(keep_all=True, post_process=False, device='cpu')
        self.resnet = InceptionResnetV1(pretrained=True).eval()
        self.faces_feature_database = get_database_faces_feature('./data/database_images')
        self.faces_name_database = [i[0] for i in self.faces_feature_database]
        self.features_database = [i[1] for i in self.faces_feature_database]

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        try:
            self.yolo_model = YOLO('./yolo/weights/best20250530.pt').to(self.device)
            print("YOLO模型加载成功，检测类别:", self.yolo_model.names)
        except Exception as e:
            print(f"YOLO加载失败: {str(e)}")
            self.yolo_model = None

        self.crossing_line_x = 320
        self.crossed_faces = set()
        self.person_objects = {}

        self.cap = cv2.VideoCapture(1)
        self.frame_width = 1280
        self.frame_height = 720
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)

    # def run(self):
    #     while self.running:
    #         ret, frame = self.cap.read()
    #         if not ret:
    #             continue
    #
    #         yolo_results = []
    #         if self.yolo_model:
    #             try:
    #                 yolo_detections = self.yolo_model.predict(source=frame, conf=0.3, imgsz=640, device=self.device, verbose=False)
    #                 if yolo_detections:
    #                     yolo_results = yolo_detections[0].boxes.data.cpu().numpy()
    #             except Exception as e:
    #                 print(f"YOLO检测出错: {str(e)}")
    #
    #         image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    #         faces, prob, bboxs, points = self.mtcnn(image, return_prob=True)
    #
    #         img_draw = Image.fromarray(image)
    #         draw = ImageDraw.Draw(img_draw)
    #
    #         draw.line([(self.crossing_line_x, 0), (self.crossing_line_x, self.frame_height)], fill=(255, 0, 0), width=5)
    #
    #         if self.yolo_model and len(yolo_results) > 0:
    #             for det in yolo_results:
    #                 x1, y1, x2, y2, conf, cls = det
    #                 label = f"{self.yolo_model.names[int(cls)]} "
    #                 draw.rectangle([x1, y1, x2, y2], outline="red", width=5)
    #                 ft = ImageFont.truetype('bgtx.ttf', size=30)
    #                 draw.text((x1, max(y1 - 30, 0)), text=label, font=ft, fill='red')
    #
    #         if faces is not None:
    #             faces_after_aligned = []
    #             for face, point in zip(faces, points):
    #                 face = face.permute(1, 2, 0).numpy().astype(np.uint8)
    #                 face_aligned, _ = Alignment(face, point)
    #                 face_aligned = torch.tensor(face_aligned, dtype=torch.float32).permute(2, 0, 1)
    #                 face_aligned = (face_aligned - 127.5) / 128
    #                 faces_after_aligned.append(face_aligned)
    #
    #             if len(faces_after_aligned) > 0:
    #                 faces_stack = torch.stack(faces_after_aligned)
    #                 faces_feature = self.resnet(faces_stack)
    #                 dist = np.array([[(e1 - e2).norm().item() for e1 in self.features_database] for e2 in faces_feature])
    #                 dist_min = dist.min(axis=1)
    #                 idx_min = dist.argmin(axis=1)
    #                 is_who = [self.faces_name_database[idx] if num < 0.7 else 'Unknown' for idx, num in zip(idx_min, dist_min)]
    #
    #                 for box, point, text in zip(bboxs, points, is_who):
    #                     rate = max(1, (box[2] - box[0]) // 100)
    #                     draw.rectangle(box.tolist(), width=2 * int(rate))
    #                     ft = ImageFont.truetype('bgtx.ttf', size=20 * int(rate))
    #                     draw.text(box[:2], text=text, font=ft, fill='blue')
    #
    #                     face_center_x = (box[0] + box[2]) // 2
    #                     current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    #
    #                     person_items = []
    #                     face_box = shapely_box(box[0], box[1], box[2], box[3])
    #
    #                     if self.yolo_model and len(yolo_results) > 0:
    #                         for det in yolo_results:
    #                             x1, y1, x2, y2, conf, cls = det
    #                             item_box = shapely_box(x1, y1, x2, y2)
    #                             if face_box.intersects(item_box):
    #                                 label = self.yolo_model.names[int(cls)]
    #                                 person_items.append(label)
    #
    #                     # 新增逻辑：检测入库/出库行为，只有拿物品才记录
    #                     if text != 'Unknown':
    #                         face_prev_pos = self.person_objects.get(f"{text}_prev_pos", None)
    #                         face_curr_pos = face_center_x
    #                         self.person_objects[f"{text}_prev_pos"] = face_curr_pos
    #
    #                         if face_prev_pos is not None:
    #                             crossed_in = face_prev_pos <= self.crossing_line_x < face_curr_pos
    #                             crossed_out = face_prev_pos >= self.crossing_line_x > face_curr_pos
    #
    #                             if crossed_in and len(person_items) > 0:
    #                                 print(f"{text} 在 {current_time} 拿了 {person_items} 入库")
    #                                 self.person_objects[text] = person_items
    #                                 self.crossed_faces.add(text)
    #
    #                             elif crossed_out and text in self.crossed_faces:
    #                                 items = self.person_objects.get(text, [])
    #                                 if len(items) > 0:
    #                                     print(f"{text} 在 {current_time} 拿了 {items} 出库")
    #                                 else:
    #                                     print(f"{text} 在 {current_time} 出库时未检测到携带物品")
    #                                 self.crossed_faces.remove(text)
    #                                 self.person_objects.pop(text, None)
    #
    #         result_frame = cv2.cvtColor(np.array(img_draw), cv2.COLOR_RGB2BGR)
    #         self.frame_data.emit(result_frame)
    #
    #     self.cap.release()
    def run(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                print("[Debug] 视频帧读取失败")
                continue

            yolo_results = []
            if self.yolo_model:
                try:
                    yolo_detections = self.yolo_model.predict(source=frame, conf=0.3, imgsz=640, device=self.device, verbose=False)
                    if yolo_detections:
                        yolo_results = yolo_detections[0].boxes.data.cpu().numpy()
                        print(f"[Debug] YOLO 检测结果数量: {len(yolo_results)}")
                except Exception as e:
                    print(f"[Error] YOLO检测出错: {str(e)}")

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            faces, prob, bboxs, points = self.mtcnn(image, return_prob=True)

            img_draw = Image.fromarray(image)
            draw = ImageDraw.Draw(img_draw)

            draw.line([(self.crossing_line_x, 0), (self.crossing_line_x, self.frame_height)], fill=(255, 0, 0), width=5)

            if self.yolo_model and len(yolo_results) > 0:
                for det in yolo_results:
                    x1, y1, x2, y2, conf, cls = det
                    label = f"{self.yolo_model.names[int(cls)]} "
                    draw.rectangle([x1, y1, x2, y2], outline="red", width=5)
                    ft = ImageFont.truetype('bgtx.ttf', size=30)
                    draw.text((x1, max(y1 - 30, 0)), text=label, font=ft, fill='red')

            if faces is not None:
                print(f"[Debug] 检测到 {len(faces)} 张人脸")
                faces_after_aligned = []
                for face, point in zip(faces, points):
                    face = face.permute(1, 2, 0).numpy().astype(np.uint8)
                    face_aligned, _ = Alignment(face, point)
                    face_aligned = torch.tensor(face_aligned, dtype=torch.float32).permute(2, 0, 1)
                    face_aligned = (face_aligned - 127.5) / 128
                    faces_after_aligned.append(face_aligned)

                if len(faces_after_aligned) > 0:
                    faces_stack = torch.stack(faces_after_aligned)
                    faces_feature = self.resnet(faces_stack)
                    dist = np.array([[(e1 - e2).norm().item() for e1 in self.features_database] for e2 in faces_feature])
                    dist_min = dist.min(axis=1)
                    idx_min = dist.argmin(axis=1)
                    is_who = [self.faces_name_database[idx] if num < 0.7 else 'Unknown' for idx, num in zip(idx_min, dist_min)]
                    print(f"[Debug] 识别结果: {is_who}")

                    for box, point, text in zip(bboxs, points, is_who):
                        rate = max(1, (box[2] - box[0]) // 100)
                        draw.rectangle(box.tolist(), width=2 * int(rate))
                        ft = ImageFont.truetype('bgtx.ttf', size=20 * int(rate))
                        draw.text(box[:2], text=text, font=ft, fill='blue')

                        face_center_x = (box[0] + box[2]) // 2
                        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

                        person_items = []
                        face_box = shapely_box(box[0], box[1], box[2], box[3])

                        if self.yolo_model and len(yolo_results) > 0:
                            for det in yolo_results:
                                x1, y1, x2, y2, conf, cls = det
                                item_box = shapely_box(x1, y1, x2, y2)
                                if face_box.intersects(item_box):
                                    label = self.yolo_model.names[int(cls)]
                                    person_items.append(label)

                        if text != 'Unknown':
                            print(f"[Debug] {text} 中心位置: {face_center_x}, 物品: {person_items}")
                            face_prev_pos = self.person_objects.get(f"{text}_prev_pos", None)
                            face_curr_pos = face_center_x
                            self.person_objects[f"{text}_prev_pos"] = face_curr_pos

                            if face_prev_pos is not None:
                                crossed_in = face_prev_pos <= self.crossing_line_x < face_curr_pos
                                crossed_out = face_prev_pos >= self.crossing_line_x > face_curr_pos

                                if crossed_in and len(person_items) > 0:
                                    print(f"{text} 在 {current_time} 拿了 {person_items} 入库")
                                    self.person_objects[text] = person_items
                                    self.crossed_faces.add(text)

                                elif crossed_out and text in self.crossed_faces:
                                    items = self.person_objects.get(text, [])
                                    if len(items) > 0:
                                        print(f"{text} 在 {current_time} 拿了 {items} 出库")
                                    else:
                                        print(f"{text} 在 {current_time} 出库时未检测到携带物品")
                                    self.crossed_faces.remove(text)
                                    self.person_objects.pop(text, None)

            result_frame = cv2.cvtColor(np.array(img_draw), cv2.COLOR_RGB2BGR)
            self.frame_data.emit(result_frame)

        self.cap.release()


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Security System")
        self.resize(1280, 720)
        self.setFixedSize(1280, 720)

        self.label = QLabel(self)
        self.label.setAlignment(Qt.AlignCenter)
        self.label.resize(1280, 720)

        qr = self.frameGeometry()
        cp = QApplication.primaryScreen().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())

        self.thread = VideoThread()
        self.thread.frame_data.connect(self.update_image)
        self.thread.start()

    def update_image(self, cv_img):
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_image)
        self.label.setPixmap(pixmap.scaled(self.label.size(), Qt.KeepAspectRatio))

    def closeEvent(self, event):
        self.thread.running = False
        self.thread.wait()
        event.accept()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec_())
