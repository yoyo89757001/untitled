import cv2


capture = cv2.VideoCapture(0)
print(capture.get(cv2.CAP_PROP_BRIGHTNESS),capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(capture.get(cv2.CAP_PROP_FPS))
i = 0
while(True):
    # 获取一帧
    ret, frame = capture.read()
    # 将这帧转换为灰度图
  # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    i += 1
    print(i)
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) == ord('q'):
        break