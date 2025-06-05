from PyQt5.QtCore import QThread, pyqtSignal
import cv2
import numpy as np
from PyQt5.QtGui import QImage
import time

class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(QImage)

    def __init__(self, process_pipeline):
        super().__init__()
        self._run_flag = True
        self.process_pipeline = process_pipeline

    import time

    def run(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("无法打开摄像头")
            return

        while self._run_flag:
            ret, frame = cap.read()
            if ret:
                processed_frame = self.process_pipeline(frame)

                rgb_image = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb_image.shape
                bytes_per_line = ch * w
                qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
                self.change_pixmap_signal.emit(qt_image)
            time.sleep(0.01)  # 加个10ms延时，减少CPU过载
        cap.release()

    def stop(self):
        self._run_flag = False
        self.wait()
