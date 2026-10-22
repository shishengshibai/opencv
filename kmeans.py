import os
import time

import cv2
import numpy as np

CAM_W, CAM_H = 640, 480
SMALL_W, SMALL_H = 160, 120      # 聚类用的缩小图
K = 4                            # 聚类数
CLUSTER_EVERY = 4                # 每 N 帧重新聚类一次（掩膜每帧都重算）

MIN_CLUSTER_RATIO = 0.006        # 簇占画面比例下限
MIN_S = 70                       # 簇的平均饱和度下限（过曝的球会掉到 70 左右）
MIN_V = 30                       # 太暗的簇不算
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

AUTO_ACQUIRE = False             # False: 先用鼠标点一下球（最稳）；按 a 可切换成自动查找
CAPTURE_DIR = "captures"         # 按 s 保存的图片放在这个文件夹，可保存任意多张


def hue_closeness(h):
    """越接近红色/橙色得分越高，H 是环形的。"""
    d = abs(h - 5)
    d = min(d, 180 - d)
    return max(0.0, 1.0 - d / 30.0)


def hue_range_mask(hsv, h_center, half_width, s_min, v_min):
    """在环形色相上取一段窗口（H 跨过 0/180 也能正确取）。"""
    lo = int(round(h_center - half_width)) % 180
    hi = int(round(h_center + half_width)) % 180
    if lo <= hi:
        return cv2.inRange(hsv, (lo, s_min, v_min), (hi, 255, 255))
    m1 = cv2.inRange(hsv, (0, s_min, v_min), (hi, 255, 255))
    m2 = cv2.inRange(hsv, (lo, s_min, v_min), (179, 255, 255))
    return cv2.bitwise_or(m1, m2)


def orange_clusters(small_bgr):
    """按 a/b 色度聚类（不看亮度），返回按“像篮球”排序的 [(score, (h,s,v))]。"""
    lab = cv2.cvtColor(small_bgr, cv2.COLOR_BGR2LAB)
    hsv = cv2.cvtColor(small_bgr, cv2.COLOR_BGR2HSV)
    ab = lab[:, :, 1:].reshape(-1, 2).astype(np.float32)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, labels, _ = cv2.kmeans(
        ab, K, None, criteria, 3, cv2.KMEANS_PP_CENTERS)
    labels = labels.reshape(lab.shape[:2])
    total = labels.size

    clusters = []
    for i in range(K):
        sel = labels == i
        n = int(sel.sum())
        if n == 0 or n / total < MIN_CLUSTER_RATIO:
            continue

        px = hsv[sel]
        s = float(px[:, 1].mean())
        v = float(px[:, 2].mean())
        ang = px[:, 0].astype(np.float32) * (np.pi / 90.0)   # H×2° 转弧度
        h = float(np.rad2deg(np.arctan2(np.sin(ang).mean(),
                                        np.cos(ang).mean()))) / 2.0 % 180.0

        if not (h <= 25 or h >= 165):       # 不够红橙
            continue
        if s < MIN_S or v < MIN_V:
            continue

        score = hue_closeness(h) * (s / 255.0) * np.sqrt(n / total)
        clusters.append((score, (h, s, v)))

    clusters.sort(key=lambda t: t[0], reverse=True)
    return clusters


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


_click_point = [None]
_last_click = [None, 999]      # 最近一次点击的位置和显示计数


def on_mouse(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        _click_point[0] = (x, y)
        _last_click[0] = (x, y)
        _last_click[1] = 0


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


def learn_from_point(frame, pt):
    """用户点一下球：学它的颜色、位置、半径，返回 (x, y, r, hue, s) 或 None。

    点在高光或阴影上也没关系：在点击周围取一块，用“饱和度较高的一半像素”
    估颜色，避免被反光/黑纹带偏。
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    height, width = frame.shape[:2]
    x = min(max(int(pt[0]), 0), width - 1)
    y = min(max(int(pt[1]), 0), height - 1)

    rad = max(12, int(0.06 * min(height, width)))
    x0, y0 = max(0, x - rad), max(0, y - rad)
    x1, y1 = min(width, x + rad + 1), min(height, y + rad + 1)
    patch = hsv[y0:y1, x0:x1].reshape(-1, 3)
    if patch.size == 0:
        return None

    bright = patch[:, 1] >= np.percentile(patch[:, 1], 55)
    if int(bright.sum()) < 10:
        bright = np.ones(len(patch), bool)
    sel = patch[bright]
    s = float(np.median(sel[:, 1]))
    ang = sel[:, 0].astype(np.float32) * (np.pi / 90.0)
    hue = float(np.rad2deg(np.arctan2(np.sin(ang).mean(),
                                      np.cos(ang).mean()))) / 2.0 % 180.0
    if s < 25:
        return None

    mask = hue_range_mask(hsv, hue, 12, max(25, int(0.6 * s)), 15)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))

    num, labels, stats, cents = cv2.connectedComponentsWithStats(mask, 8)
    label = 0
    if labels[y, x] != 0:
        label = int(labels[y, x])
    else:
        # 点在缝隙/边缘上：找包围盒包含点击点的块，否则找最近的
        best_d = float("inf")
        for i in range(1, num):
            bx, by, bw, bh = (stats[i, cv2.CC_STAT_LEFT],
                              stats[i, cv2.CC_STAT_TOP],
                              stats[i, cv2.CC_STAT_WIDTH],
                              stats[i, cv2.CC_STAT_HEIGHT])
            if bx - 5 <= x <= bx + bw + 5 and by - 5 <= y <= by + bh + 5:
                d = float(np.hypot(cents[i][0] - x, cents[i][1] - y))
                if d < best_d:
                    best_d, label = d, i
    if label == 0:
        return None

    comp = np.uint8(labels == label) * 255
    # RANSAC 拟合圆：球和手连成一整块时，也能挑出球的那段圆弧
    contours, _ = cv2.findContours(
        comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    cnt = max(contours, key=cv2.contourArea)
    height, width = frame.shape[:2]
    fit = ransac_circle(cnt.reshape(-1, 2),
                        r_min=0.04 * min(height, width),
                        r_max=0.55 * min(height, width))
    if fit is not None:
        cx, cy, radius = fit[0], fit[1], fit[2]
    else:
        (cx, cy), radius = cv2.minEnclosingCircle(cnt)

    return float(cx), float(cy), float(radius), hue, s


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

    # 锁定过球就用学到的颜色，否则用聚类结果
    if known_hue is not None:
        # 饱和度也要卡住：皮肤比球淡，卡住以后球和手/脸就不会连成一整块
        s_min = max(30, int(s_ratio * known_sat)) if known_sat else 25
        ball_mask = hue_range_mask(hsv, known_hue, 10, s_min, 20)
    else:
        ball_mask = ball_color_mask(hsv, clusters)
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


def main():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)
    if not cap.isOpened():
        print("打不开摄像头（换 cv2.VideoCapture(0) 或检查是否被其他程序占用）")
        return

    tracker = Tracker()
    clusters = []
    frame_count = 0
    auto_acquire = AUTO_ACQUIRE
    template = None

    window = "Basketball Detection"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window, on_mouse)
    print("用鼠标点一下球就能锁定；a=自动查找开关  r=重置  "
          f"s=存图(可存多张，存在 {CAPTURE_DIR}/)  q=退出")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        raw_frame = frame.copy()      # 存图用（不含画上去的框和文字）
        frame_count += 1

        # 聚类只在缩小图上做，而且不必每帧做
        if frame_count % CLUSTER_EVERY == 1 or not clusters:
            small = cv2.resize(frame, (SMALL_W, SMALL_H),
                               interpolation=cv2.INTER_AREA)
            clusters = orange_clusters(small)

        # 鼠标点了球：学颜色/位置/大小，直接锁定
        if _click_point[0] is not None:
            lock = learn_from_point(frame, _click_point[0])
            if lock is None:
                print("这个点学不到球色，点在球的中间再试一次")
            else:
                tracker.ball = [lock[0], lock[1], lock[2]]
                tracker.vel = [0.0, 0.0, 0.0]
                tracker.hue = lock[3]
                tracker.sat = lock[4]
                tracker.base_radius = lock[2]
                tracker.misses = 0
                template = make_template(
                    cv2.cvtColor(raw_frame, cv2.COLOR_BGR2GRAY),
                    lock[0], lock[1], lock[2])
                print(f"已锁定: 中心({lock[0]:.0f},{lock[1]:.0f}) "
                      f"半径 {lock[2]:.0f} H={lock[3]:.0f} S={lock[4]:.0f}")
            _click_point[0] = None

        # 没锁定又没开自动查找时，不去猜（避免框到脸）
        if tracker.ball is not None or auto_acquire:
            found, mask, match_score = detect(
                frame, clusters, tracker.ball, tracker.hue,
                tracker.base_radius, tracker.sat, template)
        else:
            found, mask = None, np.zeros(frame.shape[:2], np.uint8)
            match_score = 0.0
        tracker.update(found)

        # 球丢了就丢掉模板；匹配得很好时隔几帧刷新一次模板（适应转动/远近变化）
        if tracker.ball is None:
            template = None
        elif found is not None and match_score > 0.85 and frame_count % 10 == 0:
            new_template = make_template(
                cv2.cvtColor(raw_frame, cv2.COLOR_BGR2GRAY),
                found[0], found[1], found[2])
            if new_template is not None:
                template = new_template

        h, w = frame.shape[:2]

        if tracker.ball is not None:
            x, y, radius = (int(round(v)) for v in tracker.ball)
            radius = max(3, radius)
            x = min(max(x, -radius), w + radius)
            y = min(max(y, -radius), h + radius)
            color = (0, 255, 0) if found is not None else (0, 165, 255)
            cv2.circle(frame, (x, y), radius, color, 3)
            cv2.circle(frame, (x, y), 4, (0, 0, 255), -1)
            if found is not None:
                _, _, _, coverage, s_mean, h_mean = found
                cv2.putText(frame,
                            f"cov={coverage:.2f} S={s_mean:.0f} H={h_mean:.0f}",
                            (max(5, x - radius), max(60, y - radius - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        if tracker.ball is None and not auto_acquire:
            status, s_color = "Click on the ball", (0, 255, 255)
        elif found is not None:
            status, s_color = "Ball Detected", (0, 255, 0)
        elif tracker.ball is not None:
            status, s_color = "Tracking...", (0, 165, 255)
        else:
            status, s_color = "Searching...", (0, 0, 255)
        cv2.putText(frame, status, (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, s_color, 2)
        if auto_acquire:
            cv2.putText(frame, "AUTO", (20, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        # 显示最近一次点击位置，方便确认点到了哪里
        if _last_click[0] is not None and _last_click[1] < 15:
            cx, cy = _last_click[0]
            cv2.drawMarker(frame, (int(cx), int(cy)), (0, 255, 255),
                           cv2.MARKER_CROSS, 16, 2)
            _last_click[1] += 1

        if h > 210 and w > 270:
            mask_small = cv2.resize(mask, (240, 180))
            mask_bgr = cv2.cvtColor(mask_small, cv2.COLOR_GRAY2BGR)
            frame[h - 190:h - 10, w - 250:w - 10] = mask_bgr
            cv2.putText(frame, "Ball Color Mask", (w - 250, h - 195),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        cv2.imshow(window, frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        if key == ord('a'):
            auto_acquire = not auto_acquire
            print("自动查找:", "开" if auto_acquire else "关")
        if key == ord('r'):
            tracker = Tracker()
            template = None
            print("已重置，点一下球重新锁定")
        if key == ord('s'):
            print("候选簇:", ", ".join(
                f"H{int(hh)} S{int(ss)} V{int(vv)}"
                for _, (hh, ss, vv) in clusters))
            paths = save_capture(raw_frame, frame, mask)
            print("已保存:", " / ".join(os.path.basename(p) for p in paths),
                  f"（在 {CAPTURE_DIR} 文件夹里，可以一直按 s 存很多张）")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
