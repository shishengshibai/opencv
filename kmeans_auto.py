"""不用鼠标点击的自动找球 + 跟踪。

颜色发现（k-means）与检测/跟踪引擎都内联在本文件内，不依赖 kmeans.py。

自动锁定的原理（每一步都可单独调）：
  1. 在 Lab 的 a/b 色度上做 k-means，把画面分成几组颜色，得到若干“球色假设”；
  2. 每个假设生成一张色相掩膜，交给「判定标准」确认是不是球；
  3. 判定标准（VERIFY_MODE）默认直接用 opencv_close.py / opencv_new_close.py 的那套：
     形态学闭开 → 找轮廓 → 面积/半径过滤 → 圆度 cov 阈值。
       close      对应 opencv_close.py      (5x5 闭 + 3x3 开, cov=轮廓面积/外接圆 > 0.25)
       new_close  对应 opencv_new_close.py  (11x11 闭+开,      cov=凸包面积/外接圆 > 0.60)
       strict     本脚本原有的多门槛版（圆内球色足、圆外无同色、圆内够鲜艳、弧线吻合）
     另外三种判定之前都先过一层形状门槛（drop_non_ball_blobs）：细长（长宽比 > 2）或
     “大而不圆”的色块直接从掩膜里删掉，避免把人/衣服圈成球。
  4. 同一个候选连续 CONFIRM_FRAMES 帧半径和颜色都对得上，才真正锁定（防止偶尔误检）；
  5. 锁定后用内联的 detect（颜色掩膜 + 模板匹配 + α-β 平滑）继续跟；
     球彻底丢掉后自动重新查找，不需要按 r。

用法:
    python kmeans_auto.py                             # 摄像头 0
    python kmeans_auto.py 1                           # 摄像头 1
    python kmeans_auto.py D:\\ball.mp4                 # 视频文件（方便离线调参）
    python kmeans_auto.py D:\\ball.mp4 --verify new_close
按键: r=重新查找  v=切换判定标准  s=存图  q=退出
窗口: Auto Basketball Detection（主图）+ Ball Color Mask（掩膜，独立窗口）
"""

import math
import os
import sys
import time

import cv2
import numpy as np

# ==========================================================
# ======= 检测 / 跟踪引擎（原 kmeans.py 内联，行为不变）=======
# ==========================================================
# 以下常量与函数从 kmeans.py 原样搬来，只是为了让本文件不再依赖它。

# 缩略图尺寸与重新聚类间隔（原来也是从 kmeans.py 导入的）
SMALL_W, SMALL_H = 160, 120
CLUSTER_EVERY = 4
BALL_H_RANGE = 10                # 球色相窗口半宽
BALL_S_MIN = 35                  # 球色饱和度下限（弱光/过曝时再低也算）
COVERAGE_MIN = 0.6               # 自动查找时，圆内球色像素占比下限（严格，防框脸）
AUTO_S_MIN = 120                 # 自动查找时，圆内平均饱和度下限（皮肤一般到不了）
LOCK_COVERAGE_MIN = 0.30         # 已锁定后放宽（球被画面切掉时占比会掉到 0.4 左右）
LOCK_S_RATIO = 0.85              # 跟随时球色饱和度下限 = 锁定时的饱和度 × 这个比例
MIN_CIRC = 0.62                  # 轮廓法的圆形度
MIN_FILL = 0.45                  # 轮廓法的填充率（球被切掉也能过）
MAX_ASPECT = 2.0                 # 轮廓法宽高比
MIN_VERTS = 6                    # 多边形顶点数下限（矩形只有 4 个）
MIN_AREA_RATIO = 0.002
MIN_RADIUS_RATIO = 0.035
MAX_RADIUS_RATIO = 0.55
HOUGH_SCALE = 0.5                # 霍夫圆在缩小图上跑，快且足够准
SMOOTH = 0.7                     # 位置平滑
LEAD = 0.4                       # 速度预测
MISS_TOL = 10                    # 连续丢失多少帧后停止显示
SIZE_TOL = 0.25                  # 跟丢期间半径允许变化比例（球不会瞬间变大变小）
DIST_TOL = 0.18                  # 跟丢期间允许跳动的距离（占画面对角线比例）
MIN_TRACK_RADIUS = 4.0           # 平滑/外推后的半径下限，防止被推成负数
CAPTURE_DIR = "captures"         # 按 s 保存的图片放在这个文件夹，可保存任意多张

def hue_range_mask(hsv, h_center, half_width, s_min, v_min):
    """在环形色相上取一段窗口（H 跨过 0/180 也能正确取）。"""
    lo = int(round(h_center - half_width)) % 180
    hi = int(round(h_center + half_width)) % 180
    if lo <= hi:
        return cv2.inRange(hsv, (lo, s_min, v_min), (hi, 255, 255))
    m1 = cv2.inRange(hsv, (0, s_min, v_min), (hi, 255, 255))
    m2 = cv2.inRange(hsv, (lo, s_min, v_min), (179, 255, 255))
    return cv2.bitwise_or(m1, m2)


def ball_color_mask(hsv, clusters):
    """用聚类学到的球色做一张宽松掩膜：用来判断“这个圆里是不是球”。"""
    parts = []
    for _, (h, s, _) in clusters[:2]:
        parts.append(hue_range_mask(
            hsv, h, BALL_H_RANGE, max(BALL_S_MIN, int(0.45 * s)), 30))
    if not parts:
        parts.append(hue_range_mask(hsv, 8, 14, BALL_S_MIN, 30))
        parts.append(hue_range_mask(hsv, 172, 10, BALL_S_MIN, 30))

    mask = parts[0]
    for p in parts[1:]:
        mask = cv2.bitwise_or(mask, p)
    return mask


def circle_quality(ball_bool, hsv, x, y, radius):
    """返回 (圆内球色像素占比, 圆内平均饱和度, 圆内环形平均色相)。"""
    h, w = ball_bool.shape
    rr = max(2, int(0.85 * radius))
    x0, y0 = max(0, int(x) - rr), max(0, int(y) - rr)
    x1, y1 = min(w, int(x) + rr + 1), min(h, int(y) + rr + 1)
    if x1 <= x0 or y1 <= y0:
        return 0.0, 0.0, 0.0

    cm = np.zeros((y1 - y0, x1 - x0), np.uint8)
    cv2.circle(cm, (int(x) - x0, int(y) - y0), rr, 255, -1)
    sel = cm > 0
    covered = ball_bool[y0:y1, x0:x1]
    coverage = float(covered[sel].mean())
    s_mean = float(hsv[y0:y1, x0:x1, 1][sel].mean())

    # 色相只在“球色像素”上统计，避免背景把人球的颜色带偏
    hue_px = hsv[y0:y1, x0:x1, 0][sel & covered]
    if hue_px.size < 20:
        hue_px = hsv[y0:y1, x0:x1, 0][sel]
    ang = hue_px.astype(np.float32) * (np.pi / 90.0)
    h_mean = float(np.rad2deg(np.arctan2(np.sin(ang).mean(),
                                         np.cos(ang).mean()))) / 2.0 % 180.0
    return coverage, s_mean, h_mean


def hough_circles(frame):
    """霍夫圆：不依赖颜色，球过曝/被切掉也能找到圆弧。"""
    small = cv2.resize(frame, None, fx=HOUGH_SCALE, fy=HOUGH_SCALE,
                       interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)
    side = min(gray.shape[:2])

    circles = cv2.HoughCircles(
        gray, cv2.HOUGH_GRADIENT, dp=1.2,
        minDist=max(30, int(0.2 * side)),
        param1=110, param2=35,
        minRadius=max(10, int(0.05 * side)),
        maxRadius=int(0.55 * side))
    if circles is None:
        return []

    scale = 1.0 / HOUGH_SCALE
    return [(float(x) * scale, float(y) * scale, float(r) * scale)
            for x, y, r in circles[0]]


def find_ball(hsv, mask):
    """轮廓法：形状够圆的话也作为候选（远处小球的霍夫可能不稳）。"""
    kernel = np.ones((7, 7), np.uint8)
    m = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(
        m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(m)
    cv2.drawContours(filled, contours, -1, 255, -1)   # 填掉球纹的黑洞

    h, w = m.shape
    min_area = MIN_AREA_RATIO * h * w
    max_area = 0.8 * h * w
    min_r = MIN_RADIUS_RATIO * min(h, w)
    max_r = MAX_RADIUS_RATIO * min(h, w)

    best = None
    best_score = 0.0
    contours, _ = cv2.findContours(
        filled, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if not min_area <= area <= max_area:
            continue

        perimeter = cv2.arcLength(cnt, True)
        if perimeter == 0:
            continue
        circularity = 4 * np.pi * area / (perimeter ** 2)
        if circularity < MIN_CIRC:
            continue

        (x, y), radius = cv2.minEnclosingCircle(cnt)
        if not min_r <= radius <= max_r:
            continue

        fill = area / (np.pi * radius * radius)
        if fill < MIN_FILL:
            continue

        rw, rh = cv2.minAreaRect(cnt)[1]
        if min(rw, rh) < 1:
            continue
        aspect = max(rw, rh) / min(rw, rh)
        if aspect > MAX_ASPECT:
            continue

        # 方块/长方形只有 4 个顶点，球哪怕被切掉一半也还有 6 个以上
        if len(cv2.approxPolyDP(cnt, 0.005 * perimeter, True)) < MIN_VERTS:
            continue

        score = circularity * 0.6 + fill * 0.4
        if score > best_score:
            best_score = score
            best = (float(x), float(y), float(radius))

    return best


def hue_distance(h1, h2):
    d = abs(h1 - h2)
    return min(d, 180 - d)


def ransac_circle(points, r_min, r_max, iters=400, tol=3.0):
    """在轮廓点上用 RANSAC 拟合圆：球和手连在一起时，圆拟合会挑出球的那段弧。"""
    pts = points.astype(np.float64)
    n = len(pts)
    if n < 30:
        return None
    if n > 600:
        pts = pts[np.random.default_rng(0).choice(n, 600, replace=False)]

    best = None
    gen = np.random.default_rng(0)
    for _ in range(iters):
        a, b, c = pts[gen.choice(len(pts), 3, replace=False)]
        u, v = b - a, c - a
        det = 2 * (u[0] * v[1] - u[1] * v[0])
        if abs(det) < 1e-9:
            continue
        uu, vv = u @ u, v @ v
        ux = (v[1] * uu - u[1] * vv) / det
        uy = (u[0] * vv - v[0] * uu) / det
        cx, cy = a[0] + ux, a[1] + uy
        radius = float(np.hypot(ux, uy))
        if not (r_min <= radius <= r_max):
            continue
        dist = np.abs(np.hypot(pts[:, 0] - cx, pts[:, 1] - cy) - radius)
        inliers = int((dist < tol).sum())
        if best is None or inliers > best[0]:
            best = (inliers, cx, cy, radius)

    if best is None:
        return None
    inliers, cx, cy, radius = best
    return cx, cy, radius, inliers / len(pts)


def make_template(gray, cx, cy, radius):
    """把球那一块抠出来当模板（球纹、logo 这些特征很独特）。"""
    h, w = gray.shape
    x0, y0 = max(0, int(cx - radius)), max(0, int(cy - radius))
    x1, y1 = min(w, int(cx + radius)), min(h, int(cy + radius))
    if x1 - x0 < 20 or y1 - y0 < 20:
        return None
    return gray[y0:y1, x0:x1].copy()


def match_template(gray, templ, prev):
    """在上一帧位置附近做模板匹配，返回 (cx, cy, 匹配分数)。"""
    h, w = gray.shape
    hx, hy = templ.shape[1] / 2.0, templ.shape[0] / 2.0
    margin = int(0.25 * hx) + 24
    rx, ry = int(hx) + margin, int(hy) + margin

    x0 = max(0, int(prev[0]) - rx)
    y0 = max(0, int(prev[1]) - ry)
    x1 = min(w, int(prev[0]) + rx)
    y1 = min(h, int(prev[1]) + ry)
    roi = gray[y0:y1, x0:x1]
    if roi.shape[0] < templ.shape[0] or roi.shape[1] < templ.shape[1]:
        return None

    res = cv2.matchTemplate(roi, templ, cv2.TM_CCOEFF_NORMED)
    _, score, _, loc = cv2.minMaxLoc(res)
    return x0 + loc[0] + hx, y0 + loc[1] + hy, float(score)


def track_locked(frame, hsv, ball_bool, prev, base_radius, template):
    """已锁定时：模板匹配跟位置（不漂），半径保持锁定值（不被手撑大）。"""
    height, width = frame.shape[:2]
    diagonal = float(np.hypot(height, width))

    # 1) 模板匹配
    if template is not None:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        out = match_template(gray, template, prev)
        if out is not None:
            cx, cy, score = out
            if score >= 0.45:
                cov, s_mean, h_mean = circle_quality(
                    ball_bool, hsv, cx, cy, base_radius)
                if cov >= LOCK_COVERAGE_MIN:
                    return ((int(cx), int(cy), int(base_radius),
                             cov, s_mean, h_mean), score)

    # 2) 模板不行就退回颜色均值漂移
    ys, xs = np.nonzero(ball_bool)
    if xs.size < 80:
        return None, 0.0

    cx, cy = float(prev[0]), float(prev[1])
    sigma = max(8.0, 0.55 * base_radius)
    for _ in range(4):
        w = np.exp(-((np.hypot(xs - cx, ys - cy) / sigma) ** 2))
        wsum = float(w.sum())
        if wsum < 1e-6:
            return None, 0.0
        nx = float((xs * w).sum() / wsum)
        ny = float((ys * w).sum() / wsum)
        moved = float(np.hypot(nx - cx, ny - cy))
        cx, cy = nx, ny
        if moved < 0.5:
            break

    if np.hypot(cx - prev[0], cy - prev[1]) > DIST_TOL * diagonal:
        return None, 0.0

    cov, s_mean, h_mean = circle_quality(ball_bool, hsv, cx, cy, base_radius)
    if cov < LOCK_COVERAGE_MIN:
        return None, 0.0
    return (int(cx), int(cy), int(base_radius), cov, s_mean, h_mean), 0.0


def detect(frame, clusters, prev, known_hue=None, base_radius=None,
           known_sat=None, template=None, s_ratio=LOCK_S_RATIO):
    """霍夫圆 + 轮廓圆一起当候选，再用“圆内球色占比”验证并打分。"""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)

    # 锁定过球就用学到的颜色，否则用聚类结果
    if known_hue is not None:
        # 饱和度也要卡住：皮肤比球淡，卡住以后球和手/脸就不会连成一整块
        s_min = max(30, int(s_ratio * known_sat)) if known_sat else 25
        ball_mask = hue_range_mask(hsv, known_hue, 10, s_min, 20)
    else:
        ball_mask = ball_color_mask(hsv, clusters)
    # 暗部再要求足够的 Lab 色度，否则阴影会和球连成一片、把圆心拽偏
    ball_mask = cv2.bitwise_and(ball_mask, chroma_mask(lab, hsv))
    ball_bool = ball_mask > 0

    candidates = list(hough_circles(frame))
    contour_ball = find_ball(hsv, ball_mask)
    if contour_ball is not None:
        candidates.append(contour_ball)

    tracked = prev is not None
    if tracked:
        found, match_score = track_locked(
            frame, hsv, ball_bool, prev, base_radius, template)
        if found is not None:
            return found, ball_mask, match_score

    cov_min = LOCK_COVERAGE_MIN if tracked else COVERAGE_MIN

    h, w = frame.shape[:2]
    min_r = MIN_RADIUS_RATIO * min(h, w)
    max_r = MAX_RADIUS_RATIO * min(h, w)

    best = None
    best_score = 0.0
    diagonal = float(np.hypot(h, w))
    for x, y, radius in candidates:
        if not min_r <= radius <= max_r:
            continue

        if tracked:
            # 已经锁住球了：半径和位置都不会突变，突变的多半是脸/背景
            ref_r = base_radius if base_radius is not None else prev[2]
            if abs(radius - ref_r) > SIZE_TOL * max(ref_r, 1.0):
                continue
            if np.hypot(x - prev[0], y - prev[1]) > DIST_TOL * diagonal:
                continue

        coverage, s_mean, h_mean = circle_quality(ball_bool, hsv, x, y, radius)
        if coverage < cov_min:            # 圆里不是球色，丢掉（墙、衣服）
            continue
        if not tracked and s_mean < AUTO_S_MIN:
            continue                      # 自动查找时只认鲜艳的球，脸太淡，排除

        prox = 0.5
        size_sim = 0.5
        if prev is not None:
            d = np.hypot(x - prev[0], y - prev[1]) / (0.25 * diagonal)
            prox = max(0.0, 1.0 - d)
            size_sim = max(0.0, 1.0 - abs(radius - prev[2]) / max(prev[2], 1.0))

        score = (coverage * 0.40 + min(s_mean / 255.0, 1.0) * 0.15
                 + prox * 0.25 + size_sim * 0.20)

        # 已经锁住过球时，要求颜色基本一致，防止跳到脸上
        if known_hue is not None:
            hue_sim = max(0.0, 1.0 - hue_distance(h_mean, known_hue) / 30.0)
            if hue_sim < 0.2:
                continue
            score = score * 0.8 + hue_sim * 0.2

        if score > best_score:
            best_score = score
            best = (int(x), int(y), int(radius), coverage, s_mean, h_mean)

    return best, ball_mask, 0.0


class Tracker:
    """alpha-beta 平滑：先按速度预测，再用观测修正，稳定又不拖尾。"""

    def __init__(self):
        self.ball = None
        self.vel = [0.0, 0.0, 0.0]
        self.misses = 0
        self.hue = None
        self.sat = None
        self.base_radius = None      # 锁定时记住的半径，防止半径一路"漂"到脸上

    def update(self, found):
        if found is not None:
            x, y, r = found[0], found[1], found[2]
            if self.ball is None:
                self.ball = [float(x), float(y), float(r)]
                self.vel = [0.0, 0.0, 0.0]
            else:
                pred = [self.ball[i] + self.vel[i] for i in range(3)]
                err = (x - pred[0], y - pred[1], r - pred[2])
                self.ball = [pred[i] + SMOOTH * err[i] for i in range(3)]
                self.vel = [self.vel[i] + LEAD * err[i] for i in range(3)]

            if self.base_radius is None:
                self.base_radius = float(r)
            else:
                self.base_radius += 0.05 * (r - self.base_radius)

            # 只有检测很可信时才更新球色，避免被背景带偏
            coverage = found[3]
            if coverage >= 0.5:
                h_new = found[5]
                if self.hue is None:
                    self.hue = h_new
                else:
                    delta = (h_new - self.hue + 90) % 180 - 90
                    self.hue = (self.hue + 0.3 * delta) % 180

                s_new = found[4]
                if self.sat is None:
                    self.sat = s_new
                else:
                    self.sat += 0.3 * (s_new - self.sat)

            self.misses = 0
            self.ball[2] = max(MIN_TRACK_RADIUS, self.ball[2])
            return

        self.misses += 1
        if self.ball is not None:
            self.ball = [self.ball[i] + self.vel[i] for i in range(3)]
            self.vel = [v * 0.8 for v in self.vel]
            # 外推时半径可能被推成负数，必须夹住（否则 cv2.circle 会报错）
            self.ball[2] = max(MIN_TRACK_RADIUS, self.ball[2])
        if self.misses > MISS_TOL:
            self.ball = None
            self.vel = [0.0, 0.0, 0.0]
        if self.misses > 4 * MISS_TOL:
            self.hue = None


def save_capture(raw_frame, vis_frame, mask):
    """按一次 s 存一组图：原图 + 带标注的图 + 掩膜。不会覆盖之前的。"""
    os.makedirs(CAPTURE_DIR, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    base = os.path.join(CAPTURE_DIR, f"snap_{stamp}")
    n = 1
    while os.path.exists(base + "_raw.jpg"):
        base = os.path.join(CAPTURE_DIR, f"snap_{stamp}_{n}")
        n += 1

    raw_path = base + "_raw.jpg"
    vis_path = base + "_frame.jpg"
    mask_path = base + "_mask.png"
    cv2.imwrite(raw_path, raw_frame)
    cv2.imwrite(vis_path, vis_frame)
    cv2.imwrite(mask_path, mask)
    return raw_path, vis_path, mask_path

K = 5                       # 颜色聚类数（比点击版多一个：自动模式画面颜色更杂）
MIN_CLUSTER_RATIO = 0.008   # 一个颜色至少要占画面这么多（球的亮暗渐变会被拆成两簇，
                            # 门槛过高会让两半都“不够大”从而漏掉整个球）
CLUSTER_S_MIN = 80          # 假设色的平均饱和度下限
CLUSTER_V_MIN = 40          # 假设色的平均亮度下限
MAX_HYPOTHESES = 3          # 每帧最多试几种球色

HYP_HALF_WIDTH = 12         # 假设色的色相窗口半宽
HYP_S_RATIO = 0.55          # 假设色的饱和度下限 = 该簇平均饱和度 × 此系数
HYP_V_MIN = 25
# Lab 色度的绝对门槛：HSV 的 S 是相对量（暗部暖色 S 也很高），光看 S 分不开
# “球”和“阴影”。暗部必须有足够色度才算球色；亮度够高的像素（过曝的球色度
# 会变低）不受这条限制。
CHROMA_MIN = 30.0
CHROMA_BRIGHT_V = 150
HUE_MATCH_TOL = 18          # 候选圆内的平均色相必须和假设色接近
MIN_INLIER_RATIO = 0.18     # RANSAC 圆拟合的内点比例下限（球和手粘一起时会低一些）
MAX_RING_COV = 0.65         # 圆外一圈还允许有多少球色像素（防止把色块的“内切圆”当成球）
MIN_DISC_FILL = 0.60        # 圆内球色像素占比下限（球纹/logo 的洞先填掉再算）
MIN_SAT_RATIO = 0.85        # 圆内亮部的中位饱和度 ≥ 该簇饱和度的这个比例
MIN_SAT_ABS = 80            # 但也不能低于这个绝对下限（暗光下的灰白物体）

RADIUS_PREF = 0.18          # 期望球半径 ≈ 短边的 18%（只参与排序，不是硬条件）
CONFIRM_FRAMES = 3          # 连续多少帧都对得上才锁定
CONFIRM_MOVE_R = 1.5        # 确认时允许的帧间位移（× 当前半径，球可以飞得很快）
CONFIRM_R_RATIO = 0.35      # 确认时允许的半径变化比例
CONFIRM_HUE_TOL = 12        # 确认时允许的色相偏差

# ==========================================================
# ============ 判定标准：直接采用两个 close 脚本 ============
# ==========================================================
# 两个脚本的阈值都是按 1280x720 写死的，这里换算成比例，任意分辨率都能用。
REF_W, REF_H = 1280, 720
REF_SHORT = min(REF_W, REF_H)

VERIFY_MODES = {
    # opencv_close.py：
    #   MIN_AREA=800, MAX_AREA_RATIO=0.15, MIN_RADIUS=50, MAX_RADIUS=350
    #   5x5 闭运算 + 3x3 开运算，cov = 轮廓面积 / 最小外接圆面积 > 0.25
    "close": dict(close_k=5, open_k=3, cov_min=0.25, cov_on="contour",
                  area_min_ratio=800 / (REF_W * REF_H),
                  area_max_ratio=0.15,
                  r_min_ratio=50 / REF_SHORT,
                  r_max_ratio=350 / REF_SHORT),
    # opencv_new_close.py：
    #   面积下限 500，上限 w*h*0.25，半径 50~350
    #   11x11 闭运算 + 11x11 开运算，cov = 凸包面积 / 最小外接圆面积 > 0.6
    "new_close": dict(close_k=11, open_k=11, cov_min=0.60, cov_on="hull",
                      area_min_ratio=500 / (REF_W * REF_H),
                      area_max_ratio=0.25,
                      r_min_ratio=50 / REF_SHORT,
                      r_max_ratio=350 / REF_SHORT),
}

VERIFY_MODE = "close"       # 默认判定标准："close" / "new_close" / "strict"
VERIFY_ORDER = ("close", "new_close", "strict")   # 按 v 键切换的顺序

# ==========================================================
# ==== 额外形状门槛：两个 close 标准挡不住“人”这类大而不圆的色块 ====
# ==========================================================
CAND_MAX_ASPECT = 2.0       # 最小外接矩形长宽比：整人约 3，球约 1
CAND_BIG_R_RATIO = 0.15     # 半径超过短边这个比例的“大块”必须更像实心圆
CAND_BIG_COV_MIN = 0.65     # 大块要求的圆度（人上半身约 0.58，球 0.77~0.99）


def color_hypotheses(small_bgr):
    """按色度聚类，返回按“像球”排序的 [(score, (h, s, v))]。"""
    lab = cv2.cvtColor(small_bgr, cv2.COLOR_BGR2LAB)
    hsv = cv2.cvtColor(small_bgr, cv2.COLOR_BGR2HSV)
    ab = lab[:, :, 1:].reshape(-1, 2).astype(np.float32)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, labels, _ = cv2.kmeans(ab, K, None, criteria, 3, cv2.KMEANS_PP_CENTERS)
    labels = labels.reshape(lab.shape[:2])
    total = labels.size

    hyps = []
    for i in range(K):
        sel = labels == i
        n = int(sel.sum())
        if n == 0 or n / total < MIN_CLUSTER_RATIO:
            continue

        px = hsv[sel]
        s = float(px[:, 1].mean())
        v = float(px[:, 2].mean())
        if s < CLUSTER_S_MIN or v < CLUSTER_V_MIN:
            continue

        ang = px[:, 0].astype(np.float32) * (np.pi / 90.0)
        h = float(np.rad2deg(np.arctan2(np.sin(ang).mean(),
                                        np.cos(ang).mean()))) / 2.0 % 180.0

        # 越鲜艳、面积越大越像球；红橙色（篮球）再加一点分
        score = (s / 255.0) * float(np.sqrt(n / total))
        if h <= 25 or h >= 165:
            score *= 1.25
        hyps.append((score, (h, s, v)))

    hyps.sort(key=lambda t: t[0], reverse=True)
    return hyps[:MAX_HYPOTHESES]


def chroma_mask(lab, hsv, chroma_min=CHROMA_MIN, bright_v=CHROMA_BRIGHT_V):
    """Lab 色度 = sqrt((a-128)^2 + (b-128)^2)，与“亮不亮”关系不大。

    暗部（V < bright_v）必须色度 ≥ chroma_min 才算球色，这样阴影、黑衣服
    这类“暗但 HSV 的 S 很高”的区域会被挡掉；亮部不受限制。
    """
    a = lab[:, :, 1].astype(np.float32) - 128.0
    b = lab[:, :, 2].astype(np.float32) - 128.0
    colored = (a * a + b * b) >= (chroma_min * chroma_min)
    bright = hsv[:, :, 2] >= bright_v
    return ((colored | bright).astype(np.uint8)) * 255


def hypothesis_mask(hsv, hyp, lab=None):
    """把一种假设色变成二值掩膜（饱和度下限随该色本身的饱和度浮动）。

    这里只用 HSV：球的亮暗过渡（渐变）在 HSV 里色相/饱和度都很稳，逐像素再
    加色度限制会把暗侧削掉、让球变成“月牙”，半径量出来偏小。是否为阴影交给
    后面的 drop_non_ball_blobs 按**整块**判断。
    """
    h, s, _ = hyp
    return hue_range_mask(hsv, h, HYP_HALF_WIDTH,
                          max(45, int(HYP_S_RATIO * s)), HYP_V_MIN)


def drop_non_ball_blobs(mask, short, lab=None, hsv=None):
    """从掩膜里删掉“明显不是球”的色块。

    三条都按**整块**判断，不动球的内部像素：
      1. 细长（长宽比 > CAND_MAX_ASPECT）：整人、长条；
      2. 大而不圆（r 大但圆度低）：衣服、墙上的色块；
      3. 整块都“暗且不彩”：阴影、暗衣服 —— 球通常有亮部，亮部色度高。
    """
    out = mask.copy()
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        (cx, cy), radius = cv2.minEnclosingCircle(cnt)
        cov = area / (math.pi * radius * radius) if radius > 0 else 0.0

        rw, rh = cv2.minAreaRect(cnt)[1]
        aspect = max(rw, rh) / min(rw, rh) if min(rw, rh) > 0 else 99.0

        if aspect > CAND_MAX_ASPECT or (radius > CAND_BIG_R_RATIO * short
                                       and cov < CAND_BIG_COV_MIN):
            cv2.drawContours(out, [cnt], -1, 0, -1)
            continue

        if lab is not None and hsv is not None:
            h, w = mask.shape[:2]
            r = max(2, int(radius))
            x0, y0 = max(0, int(cx) - r), max(0, int(cy) - r)
            x1, y1 = min(w, int(cx) + r + 1), min(h, int(cy) + r + 1)
            if x1 <= x0 or y1 <= y0:
                continue
            sel = out[y0:y1, x0:x1] > 0
            if not sel.any():
                continue
            a = lab[y0:y1, x0:x1, 1].astype(np.float32) - 128.0
            b = lab[y0:y1, x0:x1, 2].astype(np.float32) - 128.0
            chroma = np.hypot(a, b)[sel]
            v_max = int(hsv[y0:y1, x0:x1, 2][sel].max())
            if float(np.percentile(chroma, 90)) < CHROMA_MIN and v_max < CHROMA_BRIGHT_V:
                cv2.drawContours(out, [cnt], -1, 0, -1)
    return out


def mask_circles(mask, min_r, max_r):
    """在色块轮廓上找圆：球和手/背景粘成一整块时，RANSAC 也能挑出球的那段弧。"""
    kernel = np.ones((7, 7), np.uint8)
    m = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(
        m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    circles = []
    min_area = 0.4 * np.pi * min_r * min_r
    for cnt in contours:
        if cv2.contourArea(cnt) < min_area:
            continue
        fit = ransac_circle(cnt.reshape(-1, 2), r_min=min_r, r_max=max_r)
        if fit is not None and fit[3] >= MIN_INLIER_RATIO:
            circles.append((fit[0], fit[1], fit[2]))
        (x, y), radius = cv2.minEnclosingCircle(cnt)
        if min_r <= radius <= max_r:
            circles.append((float(x), float(y), float(radius)))
    return circles


def disc_fill(ball_bool, x, y, radius):
    """整个圆内有多少球色像素：真球接近 1（球纹、反光会挖掉一点），
    而“盖在色块上的圆”常常只占一部分。"""
    h, w = ball_bool.shape
    r = max(2, int(round(radius)))
    x0, y0 = max(0, int(x) - r), max(0, int(y) - r)
    x1, y1 = min(w, int(x) + r + 1), min(h, int(y) + r + 1)
    if x1 <= x0 or y1 <= y0:
        return 0.0

    cm = np.zeros((y1 - y0, x1 - x0), np.uint8)
    cv2.circle(cm, (int(x) - x0, int(y) - y0), r, 255, -1)
    sel = cm > 0
    return float(ball_bool[y0:y1, x0:x1][sel].mean())


def ring_coverage(ball_bool, x, y, radius):
    """圆外一圈（1.0~1.4 倍半径）里还有多少球色像素。

    真球外面是背景；而“色块里的内切圆”外面仍然是同色，这一项能把它筛掉。
    """
    h, w = ball_bool.shape
    r_out = int(round(1.4 * radius))
    x0, y0 = max(0, int(x) - r_out), max(0, int(y) - r_out)
    x1, y1 = min(w, int(x) + r_out + 1), min(h, int(y) + r_out + 1)
    if x1 <= x0 or y1 <= y0:
        return 1.0

    ring = np.zeros((y1 - y0, x1 - x0), np.uint8)
    center = (int(x) - x0, int(y) - y0)
    cv2.circle(ring, center, r_out, 255, -1)
    cv2.circle(ring, center, max(1, int(round(radius))), 0, -1)
    sel = ring > 0
    if not sel.any():
        return 1.0
    return float(ball_bool[y0:y1, x0:x1][sel].mean())


def fill_holes(mask):
    """把掩膜里被色块包住的洞填上（球纹、logo 文字、反光会把球挖出窟窿）。"""
    filled = mask.copy()
    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(filled, contours, -1, 255, -1)
    return filled


def lit_saturation(hsv, x, y, radius):
    """圆内“亮部”的饱和度中位数：避开黑纹和阴影，比平均值可靠。

    篮球的饱和度通常接近它自己那一簇的饱和度，而皮肤、浅色衣服只有它的
    六到八成，球和脸/衣服就靠这一项分开。
    """
    h, w = hsv.shape[:2]
    r = max(2, int(0.85 * radius))
    x0, y0 = max(0, int(x) - r), max(0, int(y) - r)
    x1, y1 = min(w, int(x) + r + 1), min(h, int(y) + r + 1)
    if x1 <= x0 or y1 <= y0:
        return 0.0

    cm = np.zeros((y1 - y0, x1 - x0), np.uint8)
    cv2.circle(cm, (int(x) - x0, int(y) - y0), r, 255, -1)
    patch = hsv[y0:y1, x0:x1]
    sel = (cm > 0) & (patch[:, :, 2] >= 80)
    if int(sel.sum()) < 10:
        return 0.0
    return float(np.median(patch[:, :, 1][sel]))


def arc_support(ball_bool, x, y, radius, samples=72):
    """圆边界上有多少比例是「里面是球色、外面不是」。

    真球的半径和球色区域的边缘对得上；色块里面的小圆、画歪的圆这一项很低。
    """
    h, w = ball_bool.shape
    ang = np.linspace(0.0, 2.0 * np.pi, samples, endpoint=False)
    xi = x + 0.90 * radius * np.cos(ang)
    yi = y + 0.90 * radius * np.sin(ang)
    xo = x + 1.08 * radius * np.cos(ang)
    yo = y + 1.08 * radius * np.sin(ang)
    valid = ((xi >= 0) & (xi < w) & (yi >= 0) & (yi < h)
             & (xo >= 0) & (xo < w) & (yo >= 0) & (yo < h))
    if int(valid.sum()) < 8:
        return 0.0
    inside = ball_bool[yi[valid].astype(int), xi[valid].astype(int)]
    outside = ball_bool[yo[valid].astype(int), xo[valid].astype(int)]
    return float((inside & ~outside).mean())


def verify_mask(mask, frame_shape, cfg):
    """按 opencv_close / opencv_new_close 的判定流程确认圆。

    形态学闭开 → 找外轮廓 → 面积与半径过滤 → 圆度 cov 阈值，取 cov 最大的一个。
    返回 ((x, y, r, cov) 或 None, 形态学处理后的掩膜)。
    """
    fh, fw = frame_shape[:2]
    short = min(fh, fw)

    m = cv2.morphologyEx(
        mask, cv2.MORPH_CLOSE, np.ones((cfg["close_k"], cfg["close_k"]), np.uint8))
    m = cv2.morphologyEx(
        m, cv2.MORPH_OPEN, np.ones((cfg["open_k"], cfg["open_k"]), np.uint8))

    area_min = cfg["area_min_ratio"] * fw * fh
    area_max = cfg["area_max_ratio"] * fw * fh
    r_min = cfg["r_min_ratio"] * short
    r_max = cfg["r_max_ratio"] * short

    best = None
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area <= area_min or area > area_max:
            continue

        (cx, cy), radius = cv2.minEnclosingCircle(cnt)
        if radius < r_min or radius > r_max:
            continue

        if cfg["cov_on"] == "hull":
            ref_area = cv2.contourArea(cv2.convexHull(cnt))   # 忽略球一侧的缺口
        else:
            ref_area = area
        circle_area = math.pi * (radius ** 2)
        cov = ref_area / circle_area if circle_area > 0 else 0.0
        if cov <= cfg["cov_min"]:
            continue

        if best is None or cov > best[3]:
            best = (float(cx), float(cy), float(radius), float(cov))
    return best, m


def auto_detect(frame, hyps, mode=VERIFY_MODE):
    """未锁定时的自动查找，返回 ((x, y, r, hue, s), mask) 或 (None, mask)。

    mode 为 "close" / "new_close" 时用两个 close 脚本的判定标准；
    为 "strict" 时用本脚本原有的多门槛判定。
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    fh, fw = frame.shape[:2]

    if mode in VERIFY_MODES:
        cfg = VERIFY_MODES[mode]
        best = None
        best_cov = 0.0
        best_mask = np.zeros((fh, fw), np.uint8)
        for _hyp_score, hyp in hyps:
            mask = drop_non_ball_blobs(hypothesis_mask(hsv, hyp, lab), min(fh, fw), lab, hsv)
            found, judged = verify_mask(mask, frame.shape, cfg)
            if found is None:
                if not best_mask.any():
                    best_mask = judged
                continue
            x, y, radius, cov = found
            if cov > best_cov:
                _, _, h_mean = circle_quality(mask > 0, hsv, x, y, radius)
                best = (int(x), int(y), int(radius), h_mean,
                        lit_saturation(hsv, x, y, radius))
                best_cov = cov
                best_mask = judged

        if best is None and hyps:
            best_mask = drop_non_ball_blobs(hypothesis_mask(hsv, hyps[0][1], lab),
                                            min(fh, fw))
        return best, best_mask

    min_r = MIN_RADIUS_RATIO * min(fh, fw)
    max_r = MAX_RADIUS_RATIO * min(fh, fw)
    hough = hough_circles(frame)     # 与颜色无关，所有假设共用一次

    best = None
    best_score = 0.0
    best_mask = np.zeros((fh, fw), np.uint8)

    for hyp_score, hyp in hyps:
        mask = drop_non_ball_blobs(hypothesis_mask(hsv, hyp, lab), min(fh, fw), lab, hsv)
        ball_bool = mask > 0
        solid = fill_holes(mask) > 0     # 填掉球纹/logo 的洞再量占比
        cands = list(hough)
        cands.extend(mask_circles(mask, min_r, max_r))
        contour_ball = find_ball(hsv, mask)
        if contour_ball is not None:
            cands.append(contour_ball)

        s_floor = max(MIN_SAT_ABS, int(MIN_SAT_RATIO * hyp[1]))
        for x, y, radius in cands:
            if not min_r <= radius <= max_r:
                continue

            coverage, _, h_mean = circle_quality(
                ball_bool, hsv, x, y, radius)
            if coverage < COVERAGE_MIN:      # 圆里大半不是这个颜色，不是球
                continue
            s_lit = lit_saturation(hsv, x, y, radius)
            if s_lit < s_floor:              # 皮肤、浅色衣服都比球淡
                continue
            if ring_coverage(solid, x, y, radius) > MAX_RING_COV:
                continue                     # 圆外还是同色 → 是色块的内切圆
            if disc_fill(solid, x, y, radius) < MIN_DISC_FILL:
                continue                     # 圆里没被这个颜色填满（只是路过色块）
            if hue_distance(h_mean, hyp[0]) > HUE_MATCH_TOL:
                continue

            size_term = min(radius / (RADIUS_PREF * min(fh, fw)), 1.0)
            score = (coverage * 0.35
                     + min(s_lit / max(hyp[1], 1.0), 1.0) * 0.20
                     + hyp_score * 0.15
                     + size_term * 0.15
                     + arc_support(solid, x, y, radius) * 0.15)
            if score > best_score:
                best_score = score
                best = (int(x), int(y), int(radius), h_mean, s_lit)
                best_mask = mask

    if best is None and hyps:
        # 没找到也给张掩膜出来，方便看“为什么没找到”
        best_mask = drop_non_ball_blobs(hypothesis_mask(hsv, hyps[0][1], lab),
                                        min(fh, fw))
    return best, best_mask


class AutoAcquirer:
    """连续多帧确认：偶尔一帧误检不会锁定，锁过之后也不会立刻重复锁。"""

    def __init__(self):
        self.cand = None
        self.count = 0

    def reset(self):
        self.cand = None
        self.count = 0

    def feed(self, det):
        if det is None:
            self.reset()
            return None

        if self.cand is not None:
            r_ref = max(self.cand[2], det[2], 1.0)
            moved = float(np.hypot(det[0] - self.cand[0],
                                   det[1] - self.cand[1]))
            # 位置会因球飞得快而跳，所以主要看半径和颜色是否一致
            same = (moved <= CONFIRM_MOVE_R * r_ref
                    and abs(det[2] - self.cand[2]) <= CONFIRM_R_RATIO * r_ref
                    and hue_distance(det[3], self.cand[3]) <= CONFIRM_HUE_TOL)
            self.count = self.count + 1 if same else 1
            self.cand = det
        else:
            self.cand = det
            self.count = 1

        if self.count < CONFIRM_FRAMES:
            return None
        self.reset()
        return det


def lock_ball(tracker, det):
    x, y, radius, hue, sat = det
    tracker.ball = [float(x), float(y), float(radius)]
    tracker.vel = [0.0, 0.0, 0.0]
    tracker.hue = float(hue) % 180.0
    tracker.sat = float(sat)
    tracker.base_radius = float(radius)
    tracker.misses = 0


def open_capture(source):
    if isinstance(source, str) and source.isdigit():
        source = int(source)
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise SystemExit(f"打不开视频源: {source}"
                         "（摄像头被占用就换索引 1/2，或直接传视频文件路径）")
    if isinstance(source, int):
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    return cap


def parse_args(argv):
    """返回 (视频源, 判定标准)，支持 `kmeans_auto.py <src> --verify new_close`。"""
    src = 0
    mode = VERIFY_MODE
    rest = list(argv[1:])
    if "--verify" in rest:
        i = rest.index("--verify")
        if i + 1 < len(rest):
            mode = rest[i + 1]
            del rest[i:i + 2]
        else:
            del rest[i]
    if mode not in VERIFY_ORDER:
        print(f"未知判定标准 {mode!r}，可选 {VERIFY_ORDER}，改用 {VERIFY_MODE}")
        mode = VERIFY_MODE
    if rest:
        src = rest[0]
    return src, mode


def main():
    source, mode = parse_args(sys.argv)
    cap = open_capture(source)

    window = "Auto Basketball Detection"
    mask_window = "Ball Color Mask"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window, 960, 720)
    cv2.namedWindow(mask_window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(mask_window, 640, 480)
    print("自动找球：无需点击。r=重新查找  v=切换判定标准  s=存图  q=退出")
    print(f"判定标准: {mode}")

    tracker = Tracker()
    acquirer = AutoAcquirer()
    hyps = []
    template = None
    frame_count = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        raw_frame = frame.copy()
        frame_count += 1

        # 聚类在缩略图上做，且不必每帧做；掩膜每帧都重算
        if frame_count % CLUSTER_EVERY == 1 or not hyps:
            small = cv2.resize(frame, (SMALL_W, SMALL_H),
                               interpolation=cv2.INTER_AREA)
            hyps = color_hypotheses(small)

        found = None
        match_score = 0.0
        if tracker.ball is None:
            det, mask = auto_detect(frame, hyps, mode)
            lock = acquirer.feed(det)
            if lock is not None:
                lock_ball(tracker, lock)
                template = make_template(
                    cv2.cvtColor(raw_frame, cv2.COLOR_BGR2GRAY),
                    lock[0], lock[1], lock[2])
                print(f"自动锁定: 中心({lock[0]},{lock[1]}) 半径 {lock[2]} "
                      f"H={lock[3]:.0f} S={lock[4]:.0f}")
        else:
            found, mask, match_score = detect(
                frame, hyps, tracker.ball, tracker.hue,
                tracker.base_radius, tracker.sat, template)
            tracker.update(found)

            # 球丢了就丢模板，交给下一帧重新查找；跟得好的时候刷新模板
            if tracker.ball is None:
                template = None
                acquirer.reset()
            elif (found is not None and match_score > 0.85
                  and frame_count % 10 == 0):
                new_template = make_template(
                    cv2.cvtColor(raw_frame, cv2.COLOR_BGR2GRAY),
                    found[0], found[1], found[2])
                if new_template is not None:
                    template = new_template

        fh, fw = frame.shape[:2]

        if tracker.ball is not None:
            x, y, radius = (int(round(v)) for v in tracker.ball)
            radius = max(3, radius)
            x = min(max(x, -radius), fw + radius)
            y = min(max(y, -radius), fh + radius)
            color = (0, 255, 0) if found is not None else (0, 165, 255)
            cv2.circle(frame, (x, y), radius, color, 3)
            cv2.circle(frame, (x, y), 4, (0, 0, 255), -1)
            if found is not None:
                _, _, _, coverage, s_mean, h_mean = found
                cv2.putText(frame,
                            f"cov={coverage:.2f} S={s_mean:.0f} H={h_mean:.0f}",
                            (max(5, x - radius), max(60, y - radius - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        if tracker.ball is None:
            status, s_color = "Searching...", (0, 0, 255)
        elif found is not None:
            status, s_color = "Ball Detected", (0, 255, 0)
        else:
            status, s_color = "Tracking...", (0, 165, 255)
        cv2.putText(frame, status, (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, s_color, 2)
        cv2.putText(frame, f"verify: {mode}", (20, fh - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

        # 把当前的球色假设列出来：没有锁定时一眼能看出是不是根本聚错了颜色
        for i, (_, (hh, ss, vv)) in enumerate(hyps):
            cv2.putText(frame, f"hyp{i}: H{int(hh)} S{int(ss)} V{int(vv)}",
                        (20, 70 + 20 * i),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        cv2.imshow(window, frame)
        if mask is not None:
            cv2.imshow(mask_window, mask)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        if key == ord('r'):
            tracker = Tracker()
            acquirer.reset()
            template = None
            print("已重置，自动重新查找")
        if key == ord('v'):
            mode = VERIFY_ORDER[(VERIFY_ORDER.index(mode) + 1) % len(VERIFY_ORDER)]
            tracker = Tracker()
            acquirer.reset()
            template = None
            print(f"判定标准切换为: {mode}")
        if key == ord('s'):
            paths = save_capture(raw_frame, frame, mask)
            print("已保存:", " / ".join(paths))

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
