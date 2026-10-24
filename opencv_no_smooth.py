import cv2
import numpy as np
import math

# ===================== 你的颜色参数 =====================
LOWER_ORANGE = np.array([0, 115, 102])
UPPER_ORANGE = np.array([25, 255, 255])

# ================= 无平滑追踪变量 =================
# 检测到就直接用检测值画圈，不做任何滤波，圈永远画在球当前所在位置
last_x, last_y, last_r = 0, 0, 0
MISSING_FRAMES_MAX = 5   # 球被挡住时，圈原地停留的帧数
missing_frames = 0


def detect_basketball(frame):
    global last_x, last_y, last_r, missing_frames

    result = frame.copy()
    h, w = frame.shape[:2]
    MAX_AREA = (w * h) * 0.25

    # 1. 颜色提取
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, LOWER_ORANGE, UPPER_ORANGE)

    # 闭运算填补篮球橙色区域中的窄黑缝，再做开运算去噪
    kernel_close = np.ones((11, 11), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)
    kernel_open = np.ones((11, 11), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open)

    found_this_frame = False
    best_circle = None

    # 2. 找轮廓
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        valid_contours = [c for c in contours if cv2.contourArea(c) > 500]

        for cnt in valid_contours:
            area = cv2.contourArea(cnt)
            if area > MAX_AREA:
                continue

            (cx, cy), radius = cv2.minEnclosingCircle(cnt)

            # 过滤掉太小的（指头）和太大的（脸）
            if radius < 50 or radius > 350:
                continue

            # 【精准确认】：算凸包和圆度
            # 凸包能忽略球左侧的缺口
            hull = cv2.convexHull(cnt)
            hull_area = cv2.contourArea(hull)

            # 用凸包面积去算圆度，会非常接近真实的球体
            circle_area = math.pi * (radius ** 2)
            cov = hull_area / circle_area

            # 只有长得很像圆的才要
            if cov > 0.6:
                # 取圆度最高的那一个
                if best_circle is None or cov > best_circle[3]:
                    best_circle = (cx, cy, radius, cov)

    # 3. 直接采用检测值（无平滑）
    if best_circle is not None:
        found_this_frame = True
        missing_frames = 0
        last_x, last_y, last_r = best_circle[0], best_circle[1], best_circle[2]

    if not found_this_frame:
        if missing_frames < MISSING_FRAMES_MAX:
            missing_frames += 1
        else:
            last_x, last_y, last_r = 0, 0, 0

    # 4. 画出绿圈
    if last_r > 0:
        draw_x, draw_y, draw_r = int(last_x), int(last_y), int(last_r)
        cv2.circle(result, (draw_x, draw_y), draw_r, (0, 255, 0), 3)
        cv2.putText(result, "Basketball", (int(draw_x - draw_r), int(draw_y - draw_r - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    return result, mask


def run_on_video(video_path=0):
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    if not cap.isOpened():
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        result, mask = detect_basketball(frame)

        cv2.imshow("Basketball Detection", result)
        cv2.imshow("Mask", mask)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_on_video(0)
