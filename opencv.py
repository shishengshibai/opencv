import cv2
import numpy as np
import math

# ==========================================================
# ==================== 1. 颜色阈值参数 ====================
# ==========================================================
# 如果脸被抠出来了（Mask里脸变白），把 S 调高（比如 160）
# 如果球上的黑色字母把球切断了，把 V 调低（比如 80）
LOWER_ORANGE = np.array([0, 115, 102])   # H最小值, S最小值, V最小值
UPPER_ORANGE = np.array([20, 255, 255])  # H最大值, S最大值, V最大值

# ==========================================================
# ==================== 2. 识别过滤参数 ====================
# ==========================================================
MIN_AREA = 800          # 【最小面积】小于这个像素面积的色块直接当成噪点忽略（防嘴唇、防小光斑）
MAX_AREA_RATIO = 0.15   # 【最大面积比例】超过画面 15% 的大色块直接扔掉（防大头/防背景大块）
MIN_RADIUS = 50         # 【最小半径】小于这个像素半径的圆不认（防鼻孔、防眼睛、防小碎块）
MAX_RADIUS = 350        # 【最大半径】大于这个半径的圆不认（防天花板上的大圆灯）

# ==========================================================
# ==================== 3. 追踪与防闪烁参数 ====================
# ==========================================================
SMOOTH_FACTOR = 0.3     # 【平滑度】越小越平滑（像挂了弹簧），越大越跟手。建议 0.2 ~ 0.5
MISSING_FRAMES_MAX = 5  # 【丢失容忍度】球被手挡住时，绿圈停留在原地的帧数（1秒大概30帧）

# 全局变量，用来记录上一次画圈的位置
last_x, last_y, last_r = 0, 0, 0
missing_frames = 0


def detect_basketball(frame):
    global last_x, last_y, last_r, missing_frames
    
    result = frame.copy()
    h, w = frame.shape[:2]
    MAX_AREA = (w * h) * MAX_AREA_RATIO 
    
    # 1. 颜色提取
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, LOWER_ORANGE, UPPER_ORANGE)
    
    # 轻微去噪
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    found_this_frame = False
    best_circle = None

    # 2. 找轮廓
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        valid_contours = [c for c in contours if cv2.contourArea(c) > MIN_AREA]
        
        for cnt in valid_contours:
            area = cv2.contourArea(cnt)
            
            if area > MAX_AREA:
                continue
                
            (cx, cy), radius = cv2.minEnclosingCircle(cnt)
            
            # 半径过滤
            if radius < MIN_RADIUS or radius > MAX_RADIUS:
                continue
                
            # 圆弧覆盖率计算
            circle_area = math.pi * (radius ** 2)
            cov = area / circle_area
            
            # ==========================================================
            # ================= 4. 圆弧匹配灵敏度 =================
            # ==========================================================
            # cov 要求越小，越容易画出圈（哪怕球缺了一大半）；要求越大，越严格
            if cov > 0.25:  
                if best_circle is None or cov > best_circle[3]:
                    best_circle = (cx, cy, radius, cov)

    # 3. 平滑追踪
    if best_circle is not None:
        found_this_frame = True
        missing_frames = 0
        cx, cy, radius, cov = best_circle
        
        if last_x == 0 and last_y == 0:
            last_x, last_y, last_r = cx, cy, radius
        else:
            last_x = SMOOTH_FACTOR * cx + (1 - SMOOTH_FACTOR) * last_x
            last_y = SMOOTH_FACTOR * cy + (1 - SMOOTH_FACTOR) * last_y
            last_r = SMOOTH_FACTOR * radius + (1 - SMOOTH_FACTOR) * last_r

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
        if not ret: break
        
        result, mask = detect_basketball(frame)
        
        cv2.imshow("Basketball Detection", result)
        cv2.imshow("Mask", mask) 
        
        if cv2.waitKey(1) & 0xFF == ord('q'): 
            break
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_on_video(0)