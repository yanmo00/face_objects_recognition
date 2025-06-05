import torch
import cv2
import numpy as np
from PIL.ImageDraw import ImageDraw


def preprocess_yolo(frame, img_size=640):
    """YOLO输入预处理"""
    img = cv2.resize(frame, (img_size, img_size))
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img).float()
    img /= 255.0  # 归一化到0-1
    return img.unsqueeze(0)

def plot_yolo_results(image, results, model, colors=None):
    """绘制YOLO检测结果"""
    draw = ImageDraw.Draw(image)
    for det in results[0]:
        if det is not None and len(det):
            x1, y1, x2, y2, conf, cls = det
            label = f"{model.names[int(cls)]} {conf:.2f}"
            # 调整坐标到原始图像尺寸
            x1 = x1 * image.width / 640
            y1 = y1 * image.height / 640
            x2 = x2 * image.width / 640
            y2 = y2 * image.height / 640
            draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
            draw.text((x1, y1), text=label, fill='red')
    return image