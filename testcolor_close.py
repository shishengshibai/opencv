"""opencv_close.py 的实时调参工具（摄像头 / 视频）。

处理流程与 opencv_close.py 完全一致，只是把写死的常量换成了滑动条：

    LOWER_ORANGE / UPPER_ORANGE   -> L - H/S/V, U - H/S/V
    MIN_AREA                      -> min_area
    MAX_AREA_RATIO                -> max_area_pct (%)
    MIN_RADIUS / MAX_RADIUS       -> min_radius / max_radius
    圆弧覆盖率 cov > 0.25         -> cov_min_pct (%)
    5x5 闭运算 + 3x3 开运算       -> close_k / open_k
    SMOOTH_FACTOR                 -> smooth_pct (%)

用法:
    python testcolor_close.py             # 摄像头 0
    python testcolor_close.py video.mp4   # 视频文件

按键:
    q / ESC  退出
    s        在终端打印当前参数（可直接粘贴回 opencv_close.py）
    r        复位为 opencv_close.py 的默认值

窗口:
    Trackbars             调参面板
    Basketball Detection  当前算法最终输出（绿圈）
    Mask                  二值掩膜，调颜色阈值时主要看这个
    Candidates            所有通过面积/半径过滤的候选（黄=cov 不达标，绿=胜出），
                          圈上是 cov 值，用来定 cov_min_pct

调参思路:
    1) 只看 Mask：把球的橙色区域调成实心白块，脸、背景尽量全黑。
    2) 看 Candidates：误检的候选看它的 cov 和半径，抬高 cov_min_pct / 半径下限，
       或抬高 min_area 去掉小碎块。
    3) 回到 Basketball Detection 确认绿圈稳定跟随。
"""

import math
import sys

import cv2
import numpy as np

TITLE_PARAMS = "Trackbars"
TITLE_RESULT = "Basketball Detection"
TITLE_MASK = "Mask"
TITLE_CANDIDATES = "Candidates"

CAPTURE_WIDTH = 1280
CAPTURE_HEIGHT = 720

MISSING_FRAMES_MAX = 5  # 与 opencv_close.py 一致，球被遮挡时绿圈停留的帧数

# 与 opencv_close.py 的常量一一对应，作为滑动条默认值
DEFAULTS = {
    "L - H": 0,
    "L - S": 115,
    "L - V": 102,
    "U - H": 20,
    "U - S": 255,
    "U - V": 255,
    "close_k": 5,
    "open_k": 3,
    "min_area": 800,
    "max_area_pct": 15,
    "min_radius": 50,
    "max_radius": 350,
    "cov_min_pct": 25,
    "smooth_pct": 100,
}

SLIDERS = [
    ("L - H", 179),
    ("L - S", 255),
    ("L - V", 255),
    ("U - H", 179),
    ("U - S", 255),
    ("U - V", 255),
    ("close_k", 31),
    ("open_k", 31),
    ("min_area", 20000),
    ("max_area_pct", 100),
    ("min_radius", 600),
    ("max_radius", 600),
    ("cov_min_pct", 100),
    ("smooth_pct", 100),
]


def odd(value):
    value = max(1, int(value))
    return value if value % 2 == 1 else value + 1


def create_trackbars():
    cv2.namedWindow(TITLE_PARAMS)
    cv2.resizeWindow(TITLE_PARAMS, 500, 520)
    for name, max_value in SLIDERS:
        cv2.createTrackbar(name, TITLE_PARAMS, DEFAULTS[name], max_value, lambda _v: None)


def reset_trackbars():
    for name, _max_value in SLIDERS:
        cv2.setTrackbarPos(name, TITLE_PARAMS, DEFAULTS[name])


def read_params():
    p = {name: cv2.getTrackbarPos(name, TITLE_PARAMS) for name, _max_value in SLIDERS}
    p["close_k"] = odd(p["close_k"])
    p["open_k"] = odd(p["open_k"])
    p["min_radius"], p["max_radius"] = sorted((p["min_radius"], p["max_radius"]))
    return p


def new_state():
    return {"x": 0.0, "y": 0.0, "r": 0.0, "cov": 0.0, "missing": 0}


def detect(frame, p, st):
    """与 opencv_close.py 的 detect_basketball 同流程，参数来自滑动条。"""
    result = frame.copy()
    debug = frame.copy()
    h, w = frame.shape[:2]
    max_area = (w * h) * p["max_area_pct"] / 100.0

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower = np.array([p["L - H"], p["L - S"], p["L - V"]], dtype=np.uint8)
    upper = np.array([p["U - H"], p["U - S"], p["U - V"]], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower, upper)

    kernel_close = np.ones((p["close_k"], p["close_k"]), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)
    kernel_open = np.ones((p["open_k"], p["open_k"]), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open)

    cov_min = p["cov_min_pct"] / 100.0
    best = None

    contours, _hier = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area <= p["min_area"] or area > max_area:
            continue

        (cx, cy), radius = cv2.minEnclosingCircle(cnt)
        if radius < p["min_radius"] or radius > p["max_radius"]:
            continue

        circle_area = math.pi * (radius ** 2)
        cov = area / circle_area if circle_area > 0 else 0.0
        passed = cov > cov_min
        color = (0, 255, 0) if passed else (0, 255, 255)
        cv2.circle(debug, (int(cx), int(cy)), int(radius), color, 1)
        cv2.putText(debug, "%.2f" % cov, (int(cx) - 12, int(cy)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        if passed and (best is None or cov > best[3]):
            best = (cx, cy, radius, cov)

    if best is not None:
        st["missing"] = 0
        cx, cy, radius, cov = best
        smooth = p["smooth_pct"] / 100.0
        if st["x"] == 0 and st["y"] == 0:
            st["x"], st["y"], st["r"] = cx, cy, radius
        else:
            st["x"] = smooth * cx + (1 - smooth) * st["x"]
            st["y"] = smooth * cy + (1 - smooth) * st["y"]
            st["r"] = smooth * radius + (1 - smooth) * st["r"]
        st["cov"] = cov
    elif st["missing"] < MISSING_FRAMES_MAX:
        st["missing"] += 1
    else:
        st["x"], st["y"], st["r"], st["cov"] = 0.0, 0.0, 0.0, 0.0

    if st["r"] > 0:
        draw_x, draw_y, draw_r = int(st["x"]), int(st["y"]), int(st["r"])
        cv2.circle(result, (draw_x, draw_y), draw_r, (0, 255, 0), 3)
        cv2.putText(result, "Basketball", (draw_x - draw_r, draw_y - draw_r - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(result, "cov=%.2f" % st["cov"], (draw_x - draw_r, draw_y + draw_r + 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    return result, mask, debug


def print_report(p):
    print("\n# ---------- 粘贴回 opencv_close.py ----------")
    print("LOWER_ORANGE = np.array([%d, %d, %d])" % (p["L - H"], p["L - S"], p["L - V"]))
    print("UPPER_ORANGE = np.array([%d, %d, %d])" % (p["U - H"], p["U - S"], p["U - V"]))
    print("MIN_AREA = %d" % p["min_area"])
    print("MAX_AREA_RATIO = %.2f" % (p["max_area_pct"] / 100.0))
    print("MIN_RADIUS = %d" % p["min_radius"])
    print("MAX_RADIUS = %d" % p["max_radius"])
    print("SMOOTH_FACTOR = %.2f" % (p["smooth_pct"] / 100.0))
    print("# 闭运算核 %dx%d，开运算核 %dx%d" % (p["close_k"], p["close_k"], p["open_k"], p["open_k"]))
    print("# 圆弧覆盖率阈值 cov > %.2f" % (p["cov_min_pct"] / 100.0))
    print("# --------------------------------------------\n")


def parse_source(argv):
    if len(argv) > 1:
        return int(argv[1]) if argv[1].isdigit() else argv[1]
    return 0


def run(source=0):
    cap = cv2.VideoCapture(source)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAPTURE_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAPTURE_HEIGHT)
    if not cap.isOpened():
        print("无法打开视频源: %s" % (source,))
        return

    create_trackbars()
    st = new_state()

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        p = read_params()
        result, mask, debug = detect(frame, p, st)

        cv2.imshow(TITLE_RESULT, result)
        cv2.imshow(TITLE_MASK, mask)
        cv2.imshow(TITLE_CANDIDATES, debug)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        if key == ord("s"):
            print_report(p)
        if key == ord("r"):
            reset_trackbars()
            st = new_state()

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run(parse_source(sys.argv))
