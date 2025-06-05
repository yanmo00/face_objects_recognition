from ultralytics import YOLO
model = YOLO('yolo/weights/best202505202.pt')
results = model('test1.jpg', save=True)
results[0].show()  # 显示检测结果