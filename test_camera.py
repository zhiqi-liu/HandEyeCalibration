import cv2

cap = cv2.VideoCapture(0)  # 0 = 第一个摄像头

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
print(cap.get(cv2.CAP_PROP_FRAME_WIDTH),cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

if not cap.isOpened():
    raise RuntimeError("无法打开摄像头")

while True:
    ret, frame = cap.read()
    if not ret:
        print("读取失败")
        break

    cv2.imshow("camera", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == 27:  # ESC 退出
        break

cap.release()
cv2.destroyAllWindows()