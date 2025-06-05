import cv2

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("摄像头无法打开")
else:
    print("摄像头已打开，按 q 退出")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("读取失败")
            break
        cv2.imshow("Camera Test", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
# 设置后立即读取确认
real_w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
real_h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
print(f"实际分辨率: {int(real_w)}x{int(real_h)}")
cap.release()
cv2.destroyAllWindows()
