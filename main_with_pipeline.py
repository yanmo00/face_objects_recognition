import os
import cv2
import torch
import numpy as np
from datetime import datetime
from ultralytics import YOLO
from models.mtcnn import MTCNN
from models.face_alignment import Alignment
from models.inception_resnet_v1 import InceptionResnetV1
from get_database_faces_feature import get_database_faces_feature
from PIL import Image, ImageDraw, ImageFont
from shapely.geometry import box as shapely_box

# ========================== 全局模型和变量初始化 ==========================
device = 'cuda' if torch.cuda.is_available() else 'cpu'
mtcnn = MTCNN(keep_all=True, post_process=False, device=device)

resnet = InceptionResnetV1(pretrained=True).eval()

faces_feature_database = get_database_faces_feature('./data/database_images')
faces_name_database = [i[0] for i in faces_feature_database]
features_database = [i[1] for i in faces_feature_database]

crossing_line_x = 320
crossed_faces = set()
person_objects = {}

try:
    yolo_model = YOLO('./yolo/weights/best202505203.pt')  # 不用 .to(device)
    print("YOLO模型加载成功，检测类别:", yolo_model.names)
except Exception as e:
    print(f"YOLO加载失败: {str(e)}")
    yolo_model = None



# ========================== 图像处理流程函数 ==========================
def process_frame_pipeline(frame):
    global crossed_faces, person_objects
    frame_height, frame_width = frame.shape[:2]
    crossing_line_start = (crossing_line_x, 0)
    crossing_line_end = (crossing_line_x, frame_height)

    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    faces, prob, bboxs, points = mtcnn(image, return_prob=True)
    img_draw = Image.fromarray(image)
    draw = ImageDraw.Draw(img_draw)

    try:
        ft = ImageFont.truetype("bgtx.ttf", size=20)
    except:
        ft = ImageFont.load_default()

    draw.line([crossing_line_start, crossing_line_end], fill=(255, 0, 0), width=5)

    # YOLO 物体检测
    yolo_results = []
    if yolo_model:
        try:
            yolo_detections = yolo_model.predict(source=frame, conf=0.6, imgsz=640, device=device, verbose=False)
            if yolo_detections:
                yolo_results = yolo_detections[0].boxes.data.cpu().numpy()
        except Exception as e:
            print(f"YOLO检测出错: {str(e)}")

    if yolo_results:
        for det in yolo_results:
            x1, y1, x2, y2, conf, cls = det
            label = f"{yolo_model.names[int(cls)]} {conf:.2f}"
            draw.rectangle([x1, y1, x2, y2], outline="red", width=5)
            draw.text((x1, max(y1 - 30, 0)), text=label, font=ft, fill='red')

    if faces is not None:
        faces_after_aligned = []
        for face, point in zip(faces, points):
            face = face.permute(1, 2, 0).numpy().astype(np.uint8)
            face_aligned, _ = Alignment(face, point)
            face_aligned = torch.tensor(face_aligned, dtype=torch.float32).permute(2, 0, 1)
            face_aligned = (face_aligned - 127.5) / 128
            faces_after_aligned.append(face_aligned)

        faces_stack = torch.stack(faces_after_aligned)
        faces_feature = resnet(faces_stack)
        dist = np.array([[(e1 - e2).norm().item() for e1 in features_database] for e2 in faces_feature])
        dist_min = dist.min(axis=1)
        idx_min = dist.argmin(axis=1)
        is_who = [faces_name_database[idx] if num < 0.7 else 'Unknown' for idx, num in zip(idx_min, dist_min)]

        for box, point, text in zip(bboxs, points, is_who):
            rate = max(1, (box[2] - box[0]) // 100)
            draw.rectangle(box.tolist(), width=2 * int(rate))
            try:
                name_font = ImageFont.truetype("bgtx.ttf", size=20 * int(rate))
            except:
                name_font = ImageFont.load_default()
            draw.text(box[:2], text=text, font=name_font, fill='blue')

            face_center_x = (box[0] + box[2]) // 2
            current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

            # 检查物体交集
            person_items = []
            face_box = shapely_box(box[0], box[1], box[2], box[3])
            if yolo_results:
                for det in yolo_results:
                    x1, y1, x2, y2, conf, cls = det
                    item_box = shapely_box(x1, y1, x2, y2)
                    if face_box.intersects(item_box):
                        person_items.append(yolo_model.names[int(cls)])

            # 越界逻辑
            if face_center_x > crossing_line_x:
                if text != 'Unknown' and text not in crossed_faces:
                    crossed_faces.add(text)
                    person_objects[text] = person_items
                    print(f"{text} 在 {current_time} 拿了 {person_items} 入库")
            elif face_center_x < crossing_line_x:
                if text in crossed_faces:
                    crossed_faces.remove(text)
                    items = person_objects.get(text, [])
                    print(f"{text} 在 {current_time} 拿了 {items} 出库")
                    person_objects.pop(text, None)

    return cv2.cvtColor(np.array(img_draw), cv2.COLOR_RGB2BGR)