import cv2
import numpy as np

def nothing(x):
    pass

# 创建一个调参窗口
cv2.namedWindow("Trackbars")
cv2.resizeWindow("Trackbars", 500, 300)

# 创建滑动条 (H 从 0-179, S 和 V 从 0-255)
cv2.createTrackbar("L - H", "Trackbars", 0, 179, nothing)
cv2.createTrackbar("L - S", "Trackbars", 80, 255, nothing)  # 饱和度下限
cv2.createTrackbar("L - V", "Trackbars", 50, 255, nothing)  # 亮度下限
cv2.createTrackbar("U - H", "Trackbars", 25, 179, nothing)
cv2.createTrackbar("U - S", "Trackbars", 255, 255, nothing)
cv2.createTrackbar("U - V", "Trackbars", 255, 255, nothing)

cap = cv2.VideoCapture(0) # 摄像头

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 获取滑动条的值
    l_h = cv2.getTrackbarPos("L - H", "Trackbars")
    l_s = cv2.getTrackbarPos("L - S", "Trackbars")
    l_v = cv2.getTrackbarPos("L - V", "Trackbars")
    u_h = cv2.getTrackbarPos("U - H", "Trackbars")
    u_s = cv2.getTrackbarPos("U - S", "Trackbars")
    u_v = cv2.getTrackbarPos("U - V", "Trackbars")

    lower_orange = np.array([l_h, l_s, l_v])
    upper_orange = np.array([u_h, u_s, u_v])

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_orange, upper_orange)

    # 形态学操作（暂时先用固定的）
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=3)

    # 找轮廓
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    result = frame.copy()

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 500: # 这里暂时把最小面积设小一点，方便你看效果
            (x, y), r = cv2.minEnclosingCircle(cnt)
            cv2.circle(result, (int(x), int(y)), int(r), (255, 0, 0), 2)

    cv2.imshow("Basketball Detection", result)
    cv2.imshow("Mask", mask)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()