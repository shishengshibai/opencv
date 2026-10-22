"""不用鼠标点击的自动找球 + 跟踪。

自动锁定的原理（每一步都可单独调）：
  1. 在 Lab 的 a/b 色度上做 k-means，把画面分成几组颜色，得到若干“球色假设”；
  2. 每个假设生成一张色相掩膜，配合霍夫圆 / 轮廓圆得到圆候选；
  3. 只保留「圆内球色足、圆外基本没同色、圆内够鲜艳、圆的弧线对得上」的候选
     （皮肤、浅色衣服、墙、色块里的“内切圆”在这一步被筛掉）；
  4. 同一个候选连续 CONFIRM_FRAMES 帧半径和颜色都对得上，才真正锁定（防止偶尔误检）；
  5. 锁定后复用 kmeans.detect 的「颜色掩膜 + 模板匹配 + α-β 平滑」继续跟；
     球彻底丢掉后自动重新查找，不需要按 r。

用法:
    python kmeans_auto.py                # 摄像头 0
    python kmeans_auto.py 1              # 摄像头 1
    python kmeans_auto.py D:\\ball.mp4    # 视频文件（方便离线调参）
按键: r=重新查找  s=存图  q=退出
"""

import sys

import cv2
import numpy as np

from kmeans import (
    CLUSTER_EVERY,
    COVERAGE_MIN,
    MAX_RADIUS_RATIO,
    MIN_RADIUS_RATIO,
    SMALL_H,
    SMALL_W,
    Tracker,
    circle_quality,
    detect,
    find_ball,
    hough_circles,
    hue_distance,
    hue_range_mask,
    make_template,
    ransac_circle,
    save_capture,
)

K = 5                       # 颜色聚类数（比点击版多一个：自动模式画面颜色更杂）
MIN_CLUSTER_RATIO = 0.012   # 一个颜色至少要占画面这么多
CLUSTER_S_MIN = 80          # 假设色的平均饱和度下限
CLUSTER_V_MIN = 40          # 假设色的平均亮度下限
MAX_HYPOTHESES = 3          # 每帧最多试几种球色

HYP_HALF_WIDTH = 12         # 假设色的色相窗口半宽
HYP_S_RATIO = 0.55          # 假设色的饱和度下限 = 该簇平均饱和度 × 此系数
HYP_V_MIN = 25
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

PREVIEW_RATIO = 0.40        # 右下角掩膜预览的宽度占画面宽度比例


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


def hypothesis_mask(hsv, hyp):
    """把一种假设色变成二值掩膜（饱和度下限随该色本身的饱和度浮动）。"""
    h, s, _ = hyp
    return hue_range_mask(hsv, h, HYP_HALF_WIDTH,
                          max(45, int(HYP_S_RATIO * s)), HYP_V_MIN)


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


def auto_detect(frame, hyps):
    """未锁定时的自动查找，返回 ((x, y, r, hue, s), mask) 或 (None, mask)。"""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    fh, fw = frame.shape[:2]
    min_r = MIN_RADIUS_RATIO * min(fh, fw)
    max_r = MAX_RADIUS_RATIO * min(fh, fw)
    hough = hough_circles(frame)     # 与颜色无关，所有假设共用一次

    best = None
    best_score = 0.0
    best_mask = np.zeros((fh, fw), np.uint8)

    for hyp_score, hyp in hyps:
        mask = hypothesis_mask(hsv, hyp)
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
        best_mask = hypothesis_mask(hsv, hyps[0][1])
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


def main():
    source = sys.argv[1] if len(sys.argv) > 1 else 0
    cap = open_capture(source)

    window = "Auto Basketball Detection"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window, 960, 720)
    print("自动找球：无需点击。r=重新查找  s=存图  q=退出")

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
            det, mask = auto_detect(frame, hyps)
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

        # 把当前的球色假设列出来：没有锁定时一眼能看出是不是根本聚错了颜色
        for i, (_, (hh, ss, vv)) in enumerate(hyps):
            cv2.putText(frame, f"hyp{i}: H{int(hh)} S{int(ss)} V{int(vv)}",
                        (20, 70 + 20 * i),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        if fh > 210 and fw > 270:
            pw = min(int(fw * PREVIEW_RATIO), fw - 20)
            ph = min(pw * 3 // 4, fh - 40)
            pw = ph * 4 // 3
            mask_small = cv2.resize(mask, (pw, ph))
            mask_bgr = cv2.cvtColor(mask_small, cv2.COLOR_GRAY2BGR)
            frame[fh - ph - 10:fh - 10, fw - pw - 10:fw - 10] = mask_bgr
            cv2.putText(frame, "Ball Color Mask", (fw - pw - 10, fh - ph - 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

        cv2.imshow(window, frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        if key == ord('r'):
            tracker = Tracker()
            acquirer.reset()
            template = None
            print("已重置，自动重新查找")
        if key == ord('s'):
            paths = save_capture(raw_frame, frame, mask)
            print("已保存:", " / ".join(paths))

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
