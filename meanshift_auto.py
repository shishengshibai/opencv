"""用 mean shift 逐帧找球（独立文件，不依赖 kmeans.py）。

每一帧都从头找一次，不使用任何上一帧的结果：找到就画圈，没找到就没有圈。

流程：
  颜色发现（k-means 粗分 + mean shift 精修）
    1. cv2.pyrMeanShiftFiltering 先在像素上做联合“空间-颜色”平滑，
       把画面压成一块块纯色平台，细小噪声被抹平、物体边界保持清晰；
    2. **粗分**：在 Lab 的 a/b 色度上做 k-means，簇数取大（K=12）。球和脸的
       a/b 只差 ~9，K 小了会把它们并成一簇，后面就再也分不开；
    3. **精修**：每个粗分簇只用**自己簇内的点**跑 mean shift，收敛到该颜色的
       密度峰。只用本簇的点，才不会把相邻的肤色簇也吸进来、导致峰又合并；
    4. 每个峰换回 HSV，按饱和度/亮度/面积过滤，按“鲜艳度 × 面积”打分取前几个。

  像素归属（ownership_map）
    每个像素只归给 a/b 平面上最近的那个色模，所以不同色模的掩膜互不重叠：
    肤色像素不会混进球色掩膜，球和脸就不会连成一整块。

  外观确认（把“肤色圆块”和“真正的球”分开）
    主判据是 Lab 色度 chroma，不是 HSV 的 S：S 是相对量，室内弱光下同一个
    红球的 S 只有 90~125，用老规则 S<115 就被判成皮肤（录屏里满屏
    “圆内太淡（像皮肤）S=99”）。chroma 只反映颜色浓淡。
    所有色度刻度都**按当前帧缩放**（scene_stats）：以画面色度 p99 为基准，
      vivid = 0.70×基准（上限 36）、skin = 0.42×基准（上限 26），
      掩膜下限 = 0.60×基准（上限 28，下限 8）。
    判定顺序按“尺度无关”排：① 圆内色度 ≥ vivid 直接过；② 又淡又没对比才判
    皮肤；③ 只要比外环背景浓 4.5 倍**且**绝对差 ≥ 4 就过（弱光/过曝都能用）；
    ④ 兜底看 S。
    实测同一段视频：球 chroma 27~46，皮肤 23~35；球/背景对比 3.8~8.1 倍，
    脸/背景 2.9~4.4 倍。换成固定阈值时，亮度砍到 0.6 倍或调亮白平衡就整段
    检测不到；改成相对刻度后，欠曝 0.4~0.8 倍、±50% 变亮、冷光、褪色 40%
    都还能稳定命中。球皮纹理（边缘密度）会被运动模糊破坏，已不再作为判据。

  抗抖动
    1. k-means 固定随机种子 —— 同样的画面永远得到同样的色模；
    2. 上一帧命中的色模优先 —— 不会在几个相近色模之间来回跳；
    3. 显示位置做短窗平滑（SMOOTH_POS / SMOOTH_R），但**丢帧时立刻清空**，
       绝不拿旧值顶着。

  找球判定（沿用 opencv_close.py / opencv_new_close.py 的标准）
    三条路给出的候选全部收集起来，按“半径优先、圆度次之”挑最大的那个
    （球是画面里最大的圆形色块；以前谁先过就返回谁，脸经常抢在球前面）：
      轮廓法  形态学闭开 → 找外轮廓 → 面积/半径过滤 → 圆度 cov 阈值
        close      5x5 闭 + 3x3 开，cov = 轮廓面积 / 最小外接圆面积 > 0.25
        new_close  11x11 闭 + 11x11 开，cov = 凸包面积 / 最小外接圆面积 > 0.60
      最大内切圆  在色掩膜上做距离变换，取最厚实的圆（球被手/手臂连成一块、
                 或被运动模糊拉长时，轮廓和霍夫都会崩，但球仍是那块里最圆的实心区）
      霍夫圆      在色掩膜上直接找圆，救回被球皮黑条纹切碎的球
    后两条是兜底，只在前面一条完全找不到候选时才跑（霍夫 8ms、距离变换 2ms，
    每帧无条件跑纯属浪费）。
    判定前会先过一层形状门槛（drop_non_ball_blobs）：细长（长宽比 > 2）或“大而不圆”
    的色块直接从掩膜里整块删掉，避免把人/衣服圈成球。圆跨出画面边界时，只看画面内
    那部分算圆度，只要还有 45% 在画面里就继续判（球进出画面很常见）。

  边界重拟合（refine_circle）—— 让圈“包住”球而不是“塞进”球
    内切圆和轮廓圆都是“从掩膜里长出来的”，球被手/前臂连住或贴着画面边缘时，
    圆心会被拉走、半径会被“脖子”卡小。这里沿 36 条射线找径向边缘点
    （梯度方向与半径方向平行，|cos| ≥ 0.72），剔除离群点后做代数最小二乘圆
    拟合 —— 球的弧段即使只剩 1/3，也足以定出整圆。
    三道闸门防止拟合跑偏：半径限 0.75~1.35 倍、圆心总位移 ≤ 0.7R；
    拟合后必须有比原候选更多的边界支撑；圆内球色填充率不得低于 0.60。

  性能（延迟从哪来）
    整条流水线最贵的三样，都已按“只在需要的地方算”重写：
      1. 检测分辨率          摄像头若给 1280x720，先在原尺寸上算要 40ms/帧；
                             统一缩到 PROCESS_WIDTH(640) 宽再检测，降到约 12~15ms，
                             圆心偏移中位只有 0.07R（圆的尺寸门槛都是相对短边的比例，
                             缩放不影响判据）。有测试需求的可以把这个常数调大。
      2. pyrMeanShiftFiltering  160x120 上 sp8/sr20 要 14~29ms → 降为 sp6/sr15
                              （实测 3 段视频检出结果不变，省约 28%）；
      3. 色度闸门            全画幅 hypot 6.4ms → 每帧一张 256x256 查表算一次
                              (0.9ms)，各色模只是 bitwise_and；
      4. a/b 归属切分        全画幅 O(色模数×像素) 50ms → 只算掩膜里的像素 (1~5ms)。
      5. 有效区域裁剪        全画幅分位数统计约 6ms/帧 → 每 CONTENT_EVERY(60) 帧算一次。
      6. 边界拟合            只对每帧半径最大的 2 个候选做（每次约 1.2ms）。
    实测 1280x720 输入、640 宽处理：整帧约 12ms（约 85fps），改前是 44ms（23fps）。
    采集端同样关键：CAP_PROP_BUFFERSIZE=1 + read_latest() 每帧丢掉积压帧，
    否则驱动队列会越排越长，画面越来越滞后于现实（这是“延迟高”最常见的成因）。

  相机黑边
    内容区和黑边的交界会被误判成球，所以先用 content_box 把黑边裁掉再检测，
    坐标再换算回整幅画面；掩膜窗口用 align_mask 贴回整幅尺寸，两个窗口像素对齐。

用法:
    python meanshift_auto.py                       # 摄像头 0
    python meanshift_auto.py 1                     # 摄像头 1
    python meanshift_auto.py D:\\ball.mp4           # 视频文件（方便离线调参）
    python meanshift_auto.py D:\\ball.mp4 --verify new_close
按键: v=切换判定标准  s=存图  q=退出
窗口: Mean Shift Basketball Detection（主图）+ Mask（掩膜，独立窗口，带同一个圈）
"""

import math
import os
import sys
import time

import cv2
import numpy as np

# ===================== 基础 =====================
SMALL_W, SMALL_H = 160, 120     # 找色模用的缩略图尺寸
CLUSTER_EVERY = 4               # 每 N 帧重新找一次色模（掩膜/反投影每帧都算）
CONTENT_EVERY = 60              # 每 N 帧重算一次“有效区域”（裁黑边，全画幅统计较贵）
PROCESS_WIDTH = 640             # 检测时的最大处理宽度（更大的画面按比例缩小）
CAPTURE_DIR = "captures"        # 按 s 保存的图片目录

# ===================== 找色模：k-means 粗分 + mean shift 精修 =====================
MS_SP = 6                       # pyrMeanShiftFiltering 空间半径（像素）
MS_SR = 15                      # pyrMeanShiftFiltering 颜色半径
# 这两个值是“速度/精度”的折中：这一步是整条流水线最贵的一环（160x120 上
# sp=8/sr=20 要 14~29ms，随画面边缘密度上涨），而它只是给 k-means 做前处理。
# (6,15) 实测在 3 段视频上与 (8,20) 的检出结果一致（0 漏检，个别帧圆心差几十像素），
# 但耗时降约 28%。再往下调（5,15 / 不开平滑）会开始出现漏检和误检。
PIXEL_STEP = 2                  # 点云抽样步长（160x120 取四分之一点，够用且快）

# 粗分：簇数要够大。球和脸的 a/b 只差 ~9（实测 H3/S148 的球 vs H2/S134 的肤色），
# K 太小会把它们并成一簇，后面再怎么精修也分不开；实测 K>=10 才能分成两簇。
KMEANS_K = 12                   # 粗分簇数
KMEANS_ATTEMPTS = 3             # k-means 重启次数（取 SSE 最小的）
KMEANS_SEED = 20260101          # 固定随机种子：否则每帧聚类结果不同，圈会一直抖
MIN_COARSE_POINTS = 24          # 粗分簇至少这么多点才值得精修

# 精修：只用“本簇自己的点”迭代，所以不会把相邻的另一簇（肤色）吸过来。
# 带宽要比球/脸的 a/b 距离（~9）小，否则精修时又会被拉回同一个峰。
MS_REFINE_BW = 7.0              # mean shift 精修带宽
MS_MAX_ITER = 20                # 单个簇最多迭代次数
MS_EPS = 0.75                   # 中心移动小于该值即认为收敛
MS_MERGE_DIST = 3.5             # 精修后中心相差小于这个值就合并（重复峰）
HYP_MERGE_HUE = 6               # 归并同一颜色的色相容差（度）
HYP_MERGE_S = 35                # 归并同一颜色的饱和度容差

# 后端的“空间平滑”仍保留：把细小噪声抹平，让色模更干净
# （它和上面的颜色聚类是两件事：这个作用在像素上，那个作用在 a/b 空间）

MIN_CLUSTER_RATIO = 0.012       # 一个色模至少要占画面这么多
CLUSTER_S_MIN = 80              # 色模平均饱和度下限
CLUSTER_V_MIN = 40              # 色模平均亮度下限
MAX_HYPOTHESES = 3              # 每帧最多试几种球色
HYP_HALF_WIDTH = 12             # 假设色的色相窗口半宽
HYP_S_RATIO = 0.55              # 假设色饱和度下限 = 该色模平均饱和度 × 此系数
HYP_V_MIN = 25

# Lab 色度的绝对门槛：HSV 的 S 是相对量（暗部暖色 S 也很高），光看 S 分不开
# “球”和“阴影”。暗部必须有足够色度才算球色；亮度够高的像素（过曝的球色度
# 会变低）不受这条限制。
CHROMA_MIN = 30.0
CHROMA_BRIGHT_V = 150

# ===================== 判定标准 =====================
REF_W, REF_H = 1280, 720        # 两个 close 脚本的参考分辨率
REF_SHORT = min(REF_W, REF_H)

VERIFY_MODES = {
    "close": dict(close_k=5, open_k=3, cov_min=0.25, cov_on="contour",
                  area_min_ratio=800 / (REF_W * REF_H), area_max_ratio=0.15,
                  r_min_ratio=50 / REF_SHORT, r_max_ratio=350 / REF_SHORT),
    "new_close": dict(close_k=11, open_k=11, cov_min=0.60, cov_on="hull",
                      area_min_ratio=500 / (REF_W * REF_H), area_max_ratio=0.25,
                      r_min_ratio=50 / REF_SHORT, r_max_ratio=350 / REF_SHORT),
}
VERIFY_MODE = "close"
VERIFY_ORDER = ("close", "new_close")

# ==========================================================
# ==== 额外形状门槛：两个 close 标准挡不住“人”这类大而不圆的色块 ====
# ==========================================================
CAND_MAX_ASPECT = 2.0       # 最小外接矩形长宽比：整人约 3，球约 1
CAND_BIG_R_RATIO = 0.15     # 半径超过短边这个比例的“大块”必须更像实心圆
CAND_BIG_COV_MIN = 0.65     # 大块要求的圆度（人上半身约 0.58，球 0.77~0.99）
REQUIRE_FULL_CIRCLE = True  # 跨出画面的圆默认不认（多半是黑边/背景伪影）
CAND_EDGE_MIN_VISIBLE = 0.45  # 但球本来就会进出画面：可见比例够就继续判
CIRCLE_MIN_FILL = 0.55      # 霍夫兜底时，圆内“球色”像素的最低占比
CAND_MIN_FILL = 0.60        # 任何候选圆内“球色”占比下限（低于此不算球）
MIN_BLOB_AREA_RATIO = 0.0004  # 小于 (短边² × 此系数) 的碎块直接丢掉（省时间、去噪）

# ==========================================================
# ==== 外观确认：把“肤色圆块”和“真正的球”分开 ====
# ==========================================================
# 主判据是 Lab 色度 chroma = hypot(a-128, b-128)，不是 HSV 的 S：
#   S 是相对量，室内弱光下同一个红球只有 90~125，老规则 S<115 会直接把球
#   判成皮肤（录屏里满屏“圆内太淡（像皮肤）S=99”就是这么来的）；
#   chroma 只反映颜色浓淡，同一段视频里球 27~46、皮肤 23~35。
# 球比周围背景红得多，所以再看“圆内色度 / 外环色度”的局部对比：
#   球 3.8~8.1 倍，脸 2.9~4.4 倍 —— 运动模糊把球糊成一团时靠它兜底。
# 球皮纹理（边缘密度）会被运动模糊毁掉（实测 0.089 → 0.012），只作辅助证据。
CHROMA_VIVID = 36.0         # 色度 ≥ 这个值 → 浓到不可能是皮肤，直接通过（亮场景上限）
CHROMA_SKIN_MAX = 26.0      # 色度 < 这个值 → 和皮肤/背景一个量级，直接否（亮场景上限）
CHROMA_CONTRAST_MIN = 4.5   # 中间地带要求：圆内色度至少是外环背景的这么多倍
CHROMA_DELTA_MIN = 4.0      # 同时要求绝对差（过曝白场里 4.5/1 → 5.5 倍，但只差 2.5）
CHROMA_MASK_FLOOR = 28.0    # 颜色掩膜里低于这个色度的像素丢掉，防止球和皮肤连成一块
CHROMA_MASK_FLOOR_RATIO = 0.60  # 掩膜下限不超过全画面最强色度的这个比例（暗场景自保）
VIVID_S = 130               # 老判据保留：S 够高也算一条独立证据

# 上面这几个色度门槛都是“绝对量”。Lab 的 a*/b* 幅值随照度一起缩，同一个球
# 亮度砍到 0.6 倍色度就从 37 掉到 24，会当场被判成皮肤（实测弱光下 0/29）。
# 所以真正用的是下面这组**相对量**：以当前帧自身的高分位色度为基准缩放，
# 再用绝对刻度当上下限兜住（亮场景退化成原来的固定阈值）。
CHROMA_REF_PCT = 99.0       # 基准 = 全画面色度的 99% 分位（≈ 场上最浓的那类颜色）
CHROMA_VIVID_RATIO = 0.70   # 鲜艳线 = 基准 × 此系数
CHROMA_VIVID_MIN = 10.0     # 暗场景下限，防止基准太小时门槛塌到 0
CHROMA_SKIN_RATIO = 0.42    # 皮肤线 = 基准 × 此系数
CHROMA_SKIN_MIN = 5.0
CHROMA_MASK_FLOOR_MIN = 8.0 # 掩膜色度下限的绝对下限（暗场景）
CHROMA_MIN_RATIO = 0.60     # 色模/色块级色度门槛同样按基准缩放
SCENE_SAMPLE_STEP = 4       # 场景基准分位数的抽样步长（1/16 的像素足够）
_CHROMA_LUT = None          # 色度闸门的 256x256 查表（按 floor 缓存）
_CHROMA_LUT_FLOOR = -1.0
CLUSTER_S_RATIO = 0.60      # 色模饱和度下限 = min(固定值, 帧内 S 高分位 × 此系数)
HEAL_K = 7                  # 判形状前补裂缝用的闭运算核（要能补上球皮的黑条纹）
SMOOTH_POS = 0.45           # 显示位置平滑的下限权重（球静止时用它压抖）
SMOOTH_R = 0.30             # 显示半径平滑的下限权重
SMOOTH_GAP_FRAC = 0.60      # 预测误差超过 半径×此系数 时，权重升到 1（不平滑，零滞后）
VEL_ALPHA = 0.65            # 速度估计的平滑系数（越大越稳、越小越跟手）

DEBUG_REJECT = True         # 找球失败时，每秒把最后一个候选的被拒原因打到终端

# ==========================================================
# ==================== 工具函数 ============================
# ==========================================================
def hue_range_mask(hsv, h_center, half_width, s_min, v_min):
    """在环形色相上取一段窗口（H 跨过 0/180 也能正确取）。"""
    lo = int(round(h_center - half_width)) % 180
    hi = int(round(h_center + half_width)) % 180
    if lo <= hi:
        return cv2.inRange(hsv, (lo, s_min, v_min), (hi, 255, 255))
    m1 = cv2.inRange(hsv, (0, s_min, v_min), (hi, 255, 255))
    m2 = cv2.inRange(hsv, (lo, s_min, v_min), (179, 255, 255))
    return cv2.bitwise_or(m1, m2)


def hue_distance(h1, h2):
    d = abs(float(h1) - float(h2)) % 180.0
    return min(d, 180.0 - d)


def align_mask(mask, frame_shape, box, cropped):
    """把（可能被裁过的）掩膜贴回整幅画面，保证两个窗口像素一一对应。

    不这样做的话，掩膜窗口显示的是裁切后的小图，而圈画在整幅画面上，
    看上去就是“圈在 A 处、掩膜在别处（黑）”。
    """
    fh, fw = frame_shape[:2]
    canvas = np.zeros((fh, fw), np.uint8)
    if mask is None:
        return canvas
    if cropped:
        x0, y0, x1, y1 = box
        sub = mask[:max(0, y1 - y0), :max(0, x1 - x0)]
        canvas[y0:y0 + sub.shape[0], x0:x0 + sub.shape[1]] = sub
    else:
        h = min(fh, mask.shape[0])
        w = min(fw, mask.shape[1])
        canvas[:h, :w] = mask[:h, :w]
    return canvas


def content_box(frame, level=20, pct=90):
    """找出画面里真正有内容的矩形区域（裁掉相机填充的黑边）。

    很多摄像头/驱动在请求的比例不是原生比例时会用黑边填充（例如 4:3 传感器
    输出 720p 时，右边 960~1280 是黑的）。黑边本身颜色不在阈值内，但它和内容
    的交界会形成一条暖色过渡带，被判成"球"。这里把黑边裁掉再检测。

    判据：一整条线的 90% 分位都接近全黑才算黑边。不能只看均值——画面右侧
    偏暗的墙、深色家具容易被误当成黑边，把球的一半裁掉。
    """
    fh, fw = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cols = np.where(np.percentile(gray, pct, axis=0) >= level)[0]
    rows = np.where(np.percentile(gray, pct, axis=1) >= level)[0]
    if cols.size == 0 or rows.size == 0:
        return 0, 0, fw, fh

    x0, x1 = int(cols[0]), int(cols[-1]) + 1
    y0, y1 = int(rows[0]), int(rows[-1]) + 1
    # 裁得太狠就认为整幅都是有内容的，避免把暗场景误裁
    if (x1 - x0) < 0.3 * fw or (y1 - y0) < 0.3 * fh:
        return 0, 0, fw, fh
    return x0, y0, x1, y1


def circular_mean_hue(h_values):
    """环形平均色相（H 是 0~179 的环，不能直接算术平均）。"""
    ang = np.asarray(h_values, np.float32) * (np.pi / 90.0)   # H×2° -> 弧度
    return float(np.rad2deg(np.arctan2(np.sin(ang).mean(),
                                       np.cos(ang).mean()))) / 2.0 % 180.0


def circle_look(hsv, gray, x, y, radius, edge_thresh=120):
    """圆内的 (中位饱和度, 边缘密度)。

    边缘密度 = 梯度幅值超过 edge_thresh 的像素占比。篮球的黑条纹和 logo 会
    带来大量强边缘；人脸/皮肤是平滑的，这一项很低。这是把“肤色圆块”和
    “真正的球”分开的关键——两者在 HSV 里只差饱和度，单靠颜色分不开。
    """
    h, w = gray.shape
    r = max(2, int(radius))
    x0, y0 = max(0, int(x) - r), max(0, int(y) - r)
    x1, y1 = min(w, int(x) + r + 1), min(h, int(y) + r + 1)
    if x1 <= x0 or y1 <= y0:
        return 0.0, 0.0
    cm = np.zeros((y1 - y0, x1 - x0), np.uint8)
    cv2.circle(cm, (int(x) - x0, int(y) - y0), r, 255, -1)
    sel = cm > 0
    if not sel.any():
        return 0.0, 0.0
    s_med = float(np.median(hsv[y0:y1, x0:x1, 1][sel]))
    patch = gray[y0:y1, x0:x1]
    mag = np.hypot(cv2.Sobel(patch, cv2.CV_32F, 1, 0, ksize=3),
                   cv2.Sobel(patch, cv2.CV_32F, 0, 1, ksize=3))
    edge = float((mag[sel] > edge_thresh).mean())
    return s_med, edge


def circle_chroma(lab, x, y, radius):
    """圆内中位色度 + 圆外一圈背景的中位色度（用来算局部对比）。

    chroma 只跟“颜色浓不浓”有关，不像 HSV 的 S 那样受亮度影响，弱光下更稳。
    """
    h, w = lab.shape[:2]
    r = max(2, int(radius))
    x0, y0 = max(0, int(x) - 2 * r), max(0, int(y) - 2 * r)
    x1, y1 = min(w, int(x) + 2 * r + 1), min(h, int(y) + 2 * r + 1)
    if x1 <= x0 or y1 <= y0:
        return 0.0, 0.0
    a = lab[y0:y1, x0:x1, 1].astype(np.int16) - 128
    b = lab[y0:y1, x0:x1, 2].astype(np.int16) - 128
    # 中位数对抽样不敏感：隔 2 像素取一个，方差远小于球/皮肤之间的差，
    # 但省掉 3/4 的 hypot 计算（大球时单次 1.8ms → 0.5ms）。
    chroma = cv2.magnitude(a[::2, ::2].astype(np.float32),
                           b[::2, ::2].astype(np.float32))
    yy, xx = np.ogrid[y0 - y:y1 - y:2, x0 - x:x1 - x:2]
    dist = np.sqrt((xx * xx + yy * yy).astype(np.float32))
    inner = dist <= 0.85 * r
    outer = (dist >= 1.25 * r) & (dist <= 2.0 * r)
    if int(inner.sum()) < 5:
        return 0.0, 0.0
    c_in = float(np.median(chroma[inner]))
    if int(outer.sum()) < 10:
        return c_in, 0.0
    return c_in, float(np.median(chroma[outer]))


def mask_bbox(mask, step=4):
    """掩膜的紧凑包围盒：先降采样再去掉孤立噪点，避免被零星像素撑满全画面。

    `cv2.boundingRect(mask)` 会把任何一个孤立像素都算进去；色相窗口的掩膜边缘
    常有一堆散点，包围盒因此退化成整幅画面（720×648），后面的“只在小区域里算”
    就完全失效了。降 4 倍再取平均，只有足够密的区域能留下。
    """
    h, w = mask.shape[:2]
    if h < step or w < step:
        x, y, bw, bh = cv2.boundingRect(mask)
        return x, y, bw, bh
    sh, sw = h // step, w // step
    small = cv2.resize(mask, (sw, sh), interpolation=cv2.INTER_AREA)
    ys, xs = np.nonzero(small >= 128)
    if xs.size == 0:
        return 0, 0, 0, 0
    pad = step
    x0 = max(0, int(xs.min()) * step - pad)
    y0 = max(0, int(ys.min()) * step - pad)
    x1 = min(w, (int(xs.max()) + 1) * step + pad)
    y1 = min(h, (int(ys.max()) + 1) * step + pad)
    return x0, y0, x1 - x0, y1 - y0


def chroma_keep_mask(lab, floor):
    """整帧的“色度够浓”掩膜（0/255），每帧只算一次，供所有色模复用。

    a/b 是 uint8，量化到 2^16 种组合查一张表，比逐像素 hypot 快一倍多；
    整帧只算一次之后，每个色模只是一次 bitwise_and（0.1ms）。
    """
    global _CHROMA_LUT_FLOOR, _CHROMA_LUT
    if _CHROMA_LUT_FLOOR != floor:
        # 表按 floor 缓存：floor 只随场景基准变化，通常整段视频都是同一个值
        c = np.hypot((np.arange(256, dtype=np.float32) - 128.0)[:, None],
                     (np.arange(256, dtype=np.float32) - 128.0)[None, :])
        _CHROMA_LUT = ((c >= floor).astype(np.uint8) * 255).ravel()
        _CHROMA_LUT_FLOOR = floor
    a = lab[:, :, 1].astype(np.uint16)
    b = lab[:, :, 2].astype(np.uint16)
    idx = (a << 8) | b
    return _CHROMA_LUT[idx]


def chroma_gate(lab, mask, floor):
    """把掩膜里色度 < floor 的像素丢掉（只在掩膜的包围盒里算）。

    全画幅算一张 chroma 图要 6.4ms（720p、百万像素内存带宽），而掩膜通常只占
    画面几个百分点；限制到包围盒后不到 1ms，结果完全一致。
    """
    if floor <= 1.0:
        return mask
    x, y, w, h = mask_bbox(mask)
    if w <= 0 or h <= 0:
        return mask
    patch = lab[y:y + h, x:x + w]
    a = patch[:, :, 1].astype(np.float32)
    b = patch[:, :, 2].astype(np.float32)
    keep = (cv2.magnitude(a - 128.0, b - 128.0) >= floor).astype(np.uint8) * 255
    out = mask.copy()
    out[y:y + h, x:x + w] = cv2.bitwise_and(mask[y:y + h, x:x + w], keep)
    return out


def scene_stats(lab):
    """当前帧的色度基准，把“绝对阈值”换算成“相对阈值”。

    换光线/换白平衡时，Lab 的 a*/b* 幅值整体缩放，同一个球的绝对色度会从 37
    掉到 17；但只要“球比周围背景浓”这件事不变，判定就还能做。这里返回
    vivid / skin / floor / blob_min 四个随场景缩放的刻度。
    """
    a = lab[:, :, 1].astype(np.int16) - 128
    b = lab[:, :, 2].astype(np.int16) - 128
    s = SCENE_SAMPLE_STEP
    # 基准取 99% 分位，对离群点不敏感；抽样算就够（1/16 抽样与全画幅差 <1%），
    # 而且完全不需要生成整张 chroma 图 —— 阈值只在掩膜包围盒里用得上。
    sub = cv2.magnitude(a[::s, ::s].astype(np.float32), b[::s, ::s].astype(np.float32))
    ref = float(np.percentile(sub, CHROMA_REF_PCT))
    clamp = lambda v, lo, hi: max(lo, min(hi, v))
    return dict(
        ref=ref,
        vivid=clamp(CHROMA_VIVID_RATIO * ref, CHROMA_VIVID_MIN, CHROMA_VIVID),
        skin=clamp(CHROMA_SKIN_RATIO * ref, CHROMA_SKIN_MIN, CHROMA_SKIN_MAX),
        floor=clamp(CHROMA_MASK_FLOOR_RATIO * ref, CHROMA_MASK_FLOOR_MIN,
                    CHROMA_MASK_FLOOR),
        blob_min=clamp(CHROMA_MIN_RATIO * ref, CHROMA_SKIN_MIN, CHROMA_MIN),
    )


def ball_look_ok(hsv, gray, lab, x, y, radius, stats=None):
    """这个圆看起来像球吗？返回 (通过, 原因)。

    判据按“尺度无关”往前排，绝对色度只作下限：
      1. 圆内色度 ≥ vivid（随场景缩放）→ 直接通过；
      2. 又淡又没对比（chroma < skin 且 比外环背景不到 CHROMA_CONTRAST_MIN 倍）
         → 判皮肤直接否；
      3. 只要比外环背景浓 CHROMA_CONTRAST_MIN 倍以上就通过 —— 这一条在暗光、
         过曝时都能用，因为它只比较“球和周围”，不看绝对值；
      4. 兜底看 S 是否足够高。
    """
    stats = stats or scene_stats(lab)
    s_med, _edge = circle_look(hsv, gray, x, y, radius)
    chroma, ring = circle_chroma(lab, x, y, radius)
    contrast = chroma / max(ring, 1.0)
    delta = chroma - ring
    redder = contrast >= CHROMA_CONTRAST_MIN and delta >= CHROMA_DELTA_MIN
    if chroma >= stats["vivid"]:
        return True, "色度够浓 %.0f（门槛 %.0f）" % (chroma, stats["vivid"])
    if chroma < stats["skin"] and not redder:
        return False, ("色度低又没对比（%.0f < %.0f，对比 %.1f 倍/差 %.1f，像皮肤）"
                       % (chroma, stats["skin"], contrast, delta))
    if redder:
        return True, "比背景红 %.1f 倍（差 %.1f）" % (contrast, delta)
    if s_med >= VIVID_S:
        return True, "饱和度高（S=%.0f）" % s_med
    return False, "色度 %.0f 对比 %.1f 倍都不够（像皮肤）" % (chroma, contrast)


def circle_stats(hsv, x, y, radius):
    """圆内亮部饱和度中位数 + 亮部环形平均色相（锁定瞬间记下球的颜色）。"""
    h, w = hsv.shape[:2]
    r = max(2, int(0.85 * radius))
    x0, y0 = max(0, int(x) - r), max(0, int(y) - r)
    x1, y1 = min(w, int(x) + r + 1), min(h, int(y) + r + 1)
    if x1 <= x0 or y1 <= y0:
        return 0.0, 0.0

    patch = hsv[y0:y1, x0:x1]
    cm = np.zeros(patch.shape[:2], np.uint8)
    cv2.circle(cm, (int(x) - x0, int(y) - y0), r, 255, -1)
    sel = (cm > 0) & (patch[:, :, 2] >= 80)      # 跳过黑纹、阴影
    if int(sel.sum()) < 10:
        sel = cm > 0
    s_med = float(np.median(patch[:, :, 1][sel]))
    h_mean = circular_mean_hue(patch[:, :, 0][sel])
    return s_med, h_mean


# ==========================================================
# ============ 找色模：k-means 粗分 → mean shift 精修 =======
# ==========================================================
def kmeans_coarse(points, k=KMEANS_K, attempts=KMEANS_ATTEMPTS):
    """粗分：在 a/b 平面上做 k-means，簇数取大一些。

    目的不是“分的准”，而是**先把可能不同的颜色拆开**：球和脸的 a/b 只差 ~9，
    小 K 会把它们并成一簇，后续任何颜色/形状判据都救不回来。K 取大，
    肤色、球色、墙色、地板色各自成簇，再由 mean shift 逐个收敛。
    """
    n = len(points)
    k = max(2, min(k, n // MIN_COARSE_POINTS))
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    cv2.setRNGSeed(KMEANS_SEED)     # 固定种子：同一帧输入永远得到同样的簇
    _, labels, centers = cv2.kmeans(points, k, None, criteria, attempts,
                                    cv2.KMEANS_PP_CENTERS)
    return labels.ravel(), centers


def mean_shift_refine(points, labels, centers, bandwidth=MS_REFINE_BW,
                      max_iter=MS_MAX_ITER, eps=MS_EPS):
    """精修：每个粗分簇**只用自己簇内的点**跑 mean shift，收敛到簇内密度峰。

    这是“粗分 + 精修”能work的关键：如果拿全部点做 mean shift，带宽内会把
    相邻的肤色簇也吸进来，两个峰又并成一个。只用本簇的点，等于在每个粗分
    区域内部重新找重心，既保留粗分的边界，又让中心落在真正的密度峰上。

    返回 [(中心 a/b, 支持点数, 该峰包含的点下标)]。
    """
    bw2 = bandwidth * bandwidth
    modes = []
    for c in range(len(centers)):
        idx = np.where(labels == c)[0]
        if len(idx) < MIN_COARSE_POINTS:
            continue
        member = points[idx]

        center = centers[c].astype(np.float32).copy()
        sel = None
        for _ in range(max_iter):
            sel = ((member - center) ** 2).sum(1) <= bw2
            if not sel.any():
                sel = np.ones(len(member), bool)      # 带宽内没点就退回整簇
                break
            new_center = member[sel].mean(0)
            shift = float(np.linalg.norm(new_center - center))
            center = new_center
            if shift < eps:
                break

        if sel is None or not sel.any():
            sel = np.ones(len(member), bool)
        support = int(sel.sum())
        peak_idx = idx[sel]

        merged = False
        for i, (mode, count, prev_idx) in enumerate(modes):
            if float(np.linalg.norm(mode - center)) < MS_MERGE_DIST:
                total = count + support
                modes[i] = ((mode * count + center * support) / total, total,
                            np.concatenate([prev_idx, peak_idx]))
                merged = True
                break
        if not merged:
            modes.append((center, support, peak_idx))
    return modes


def merge_modes_by_color(modes, hsv_sampled):
    """把精修出来的峰按 HSV 相近归并回“一种颜色”。

    k-means 大 K 会把同一颗球沿亮暗渐变切成好几个小簇（实测一个渐变球被切成
    7 个峰，每个只占画面 0.35%），单个簇的占比都达不到门槛。它们其实是一种
    颜色（H13~14 / S≈196），所以这里把色相、饱和度相近的峰合并，面积累加。

    返回 [(中心 a/b, 支持点数, 点下标, (h, s, v))]。
    """
    items = []
    for center, support, idx in modes:
        px = hsv_sampled[idx]
        if len(px) < 8:
            continue
        items.append([center, support, idx,
                      (circular_mean_hue(px[:, 0]),
                       float(px[:, 1].mean()), float(px[:, 2].mean()))])
    items.sort(key=lambda t: -t[1])

    merged = []
    for center, support, idx, hsv_triple in items:
        h, s, _v = hsv_triple
        for m in merged:
            mh, ms_, _mv = m[3]
            if (hue_distance(h, mh) <= HYP_MERGE_HUE
                    and abs(s - ms_) <= HYP_MERGE_S):
                total = m[1] + support
                m[0] = (m[0] * m[1] + center * support) / total
                m[1] = total
                m[2] = np.concatenate([m[2], idx])
                px = hsv_sampled[m[2]]
                m[3] = (circular_mean_hue(px[:, 0]),
                        float(px[:, 1].mean()), float(px[:, 2].mean()))
                break
        else:
            merged.append([center, support, idx, hsv_triple])
    return [(m[0], m[1], m[2], m[3]) for m in merged]


def color_hypotheses(small_bgr):
    """返回 (hyps, centers)。

    hyps    = [(score, (h, s, v), mode_idx), ...]，按“像球”排序，最多
              MAX_HYPOTHESES 个；mode_idx 是它在 centers 里的下标。
    centers = (M, 2) 的 a/b 中心，包含**所有**通过过滤的色模（不只是前几个）。
              用来做“像素归属划分”：每个像素只归给最近的那个色模，这样不同
              色模的掩膜不会重叠（肤色像素不会混进球色掩膜）。

    三步：k-means 大 K 粗分（把球/脸/墙/地板拆开）→ 每个簇内 mean shift 精修
    （把簇心移到该颜色真正的密度峰）→ 按 HSV 相近归并（把同一颗球被渐变切出
    的碎片合回去），再按饱和度/亮度/色度/面积过滤打分。
    """
    filtered = cv2.pyrMeanShiftFiltering(small_bgr, MS_SP, MS_SR)

    lab = cv2.cvtColor(filtered, cv2.COLOR_BGR2LAB)
    hsv = cv2.cvtColor(filtered, cv2.COLOR_BGR2HSV)

    ab = lab[:, :, 1:3].reshape(-1, 2).astype(np.float32)
    sampled = ab[::PIXEL_STEP]
    if len(sampled) < MIN_COARSE_POINTS * 2:
        return [], None
    hsv_sampled = hsv.reshape(-1, 3)[::PIXEL_STEP]

    labels, centers = kmeans_coarse(sampled)
    modes = merge_modes_by_color(mean_shift_refine(sampled, labels, centers),
                                 hsv_sampled)

    total = float(len(sampled))
    # 归属划分用**全部**色模（不只是通过过滤的）：肤色/墙色即使因为不够鲜艳
    # 当不成“球色假设”，也要在 a/b 平面上占住自己的地盘，否则球的掩膜会把
    # 它们一并圈进去。
    all_centers = (np.array([m[0] for m in modes], np.float32)
                   if modes else None)

    kept = []            # [(score, (h, s, v), center)]
    # 帧内分布：S 的高分位代表“这个画面里最鲜艳的颜色”，色度同理。
    s_hi = float(np.percentile(hsv_sampled[:, 1], 95))
    ab_c = np.hypot(sampled[:, 0] - 128.0, sampled[:, 1] - 128.0)
    c_hi = float(np.percentile(ab_c, CHROMA_REF_PCT))
    s_cut = min(CLUSTER_S_MIN, CLUSTER_S_RATIO * s_hi)
    chroma_cut = min(CHROMA_MIN, CHROMA_MIN_RATIO * c_hi)
    for center, support, peak_idx, (h, s, v) in modes:
        ratio = support / total
        if ratio < MIN_CLUSTER_RATIO:
            continue

        if support < 16:
            continue
        # 饱和度/色度门槛都按本帧自身分布缩放：画面整体变亮或变暗时，球的
        # 绝对 S 会从 103 掉到 57，用固定 80 的话色模连生成的机会都没有。
        if s < s_cut or v < CLUSTER_V_MIN:
            continue

        # 暗而色度低的色模是阴影/暗衣服，不是球（色模中心就在 a/b 平面上，直接算色度）
        chroma = float(np.hypot(center[0] - 128.0, center[1] - 128.0))
        if chroma < chroma_cut and v < CHROMA_BRIGHT_V:
            continue

        score = (s / 255.0) * float(np.sqrt(ratio))
        if h <= 25 or h >= 165:       # 红橙色（篮球）加成
            score *= 1.25
        kept.append((score, (h, s, v), center))

    if not kept:
        return [], all_centers

    kept.sort(key=lambda t: t[0], reverse=True)
    hyps = []
    for score, hsv_triple, center in kept[:MAX_HYPOTHESES]:
        idx = int(np.argmin(((all_centers - center) ** 2).sum(1)))
        hyps.append((score, hsv_triple, idx))
    return hyps, all_centers


def split_by_owner(mask, lab, centers, mode_idx):
    """只保留“归属给第 mode_idx 个色模”的像素（按 a/b 最近邻归属）。

    与旧写的区别：只在**掩膜已经选中的像素**上算距离（通常只占画面 10%），
    而不是给整幅画面每个像素都和所有色模比一遍。720p + 10 个色模时，
    全画面版要 35ms，这里不到 1ms，结果逐像素一致。
    """
    if centers is None or len(centers) == 0 or mode_idx is None:
        return mask
    ys, xs = np.nonzero(mask)
    if ys.size == 0:
        return mask
    a = lab[ys, xs, 1].astype(np.int32)
    b = lab[ys, xs, 2].astype(np.int32)
    best = np.full(ys.shape, np.iinfo(np.int32).max, np.int32)
    owner = np.zeros(ys.shape, np.int32)
    for i, c in enumerate(centers):
        da = a - int(c[0])
        db = b - int(c[1])
        d = da * da + db * db
        closer = d < best
        best[closer] = d[closer]
        owner[closer] = i
    keep = owner == mode_idx
    out = np.zeros_like(mask)
    out[ys[keep], xs[keep]] = 255
    return out


def hypothesis_mask(hsv, hyp, centers=None, mode_idx=None, lab=None):
    """把一种假设色变成掩膜（饱和度下限随该色本身的饱和度浮动）。

    只用 HSV：球的亮暗过渡（渐变）在色相/饱和度上都很稳，逐像素再加色度限制
    会把暗侧削掉、让球变成“月牙”。是否为阴影交给 drop_non_ball_blobs 按整块判断。

    给了 centers/mode_idx（且带了 lab）时，只保留“归属给这个色模”的像素
    （见 split_by_owner），于是同一颗球的掩膜里不会混进肤色/墙色这些相邻
    色模的像素。
    """
    h, s, _ = hyp
    mask = hue_range_mask(hsv, h, HYP_HALF_WIDTH,
                          max(45, int(HYP_S_RATIO * s)), HYP_V_MIN)
    if lab is not None and centers is not None and mode_idx is not None:
        mask = split_by_owner(mask, lab, centers, mode_idx)
    return mask


def drop_non_ball_blobs(mask, short, lab=None, hsv=None, chroma_min=CHROMA_MIN):
    """从掩膜里删掉“明显不是球”的色块。

    三条都按**整块**判断，不动球的内部像素：
      1. 细长（长宽比 > CAND_MAX_ASPECT）：整人、长条；
      2. 大而不圆（r 大但圆度低）：衣服、墙上的色块；
      3. 整块都“暗且不彩”：阴影、暗衣服 —— 球通常有亮部，亮部色度高。
    chroma_min 由调用方按当前帧的色度基准传进来（暗光下自动调低）。
    """
    out = mask.copy()
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_area = max(4.0, MIN_BLOB_AREA_RATIO * short * short)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            # 1~2 像素的碎块：既不是球，也没必要走后面的距离变换/形态学
            cv2.drawContours(out, [cnt], -1, 0, -1)
            continue
        (cx, cy), radius = cv2.minEnclosingCircle(cnt)
        cov = area / (math.pi * radius * radius) if radius > 0 else 0.0

        rw, rh = cv2.minAreaRect(cnt)[1]
        aspect = max(rw, rh) / min(rw, rh) if min(rw, rh) > 0 else 99.0

        if aspect > CAND_MAX_ASPECT or (radius > CAND_BIG_R_RATIO * short
                                       and cov < CAND_BIG_COV_MIN):
            # 细长/大而不圆：整块不能当球。
            # 注意这里**不**从块里“切一个圆核”出来：球和手同色连成一片时，
            # 切出来的圆核确实像球，但同样也会从“人 + 衣服”这种大块里切出
            # 脸/肩膀那么大的圆核，白白多出误检（实测在没有人球的画面里
            # 多出 3 帧假圈）。球和手粘连的情况交给 inscribed_circles 兜底，
            # 它只在这一路完全找不到候选时才跑。
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
            if float(np.percentile(chroma, 90)) < chroma_min and v_max < CHROMA_BRIGHT_V:
                cv2.drawContours(out, [cnt], -1, 0, -1)
    return out


# ==========================================================
# ==================== 判定（close / new_close）=============
# ==========================================================
def disc_visibility(cx, cy, radius, fw, fh):
    """圆落在画面内的比例（0~1）。球进出画面时用得上。"""
    r = int(round(radius))
    if r <= 0:
        return 1.0
    m = np.zeros((2 * r + 1, 2 * r + 1), np.uint8)
    cv2.circle(m, (r, r), r, 255, -1)
    total = int((m > 0).sum())
    if total == 0:
        return 0.0
    x0, y0 = int(round(cx)) - r, int(round(cy)) - r
    sx0, sy0 = max(0, -x0), max(0, -y0)
    sx1 = m.shape[1] - max(0, x0 + m.shape[1] - fw)
    sy1 = m.shape[0] - max(0, y0 + m.shape[0] - fh)
    if sx1 <= sx0 or sy1 <= sy0:
        return 0.0
    return float((m[sy0:sy1, sx0:sx1] > 0).sum()) / total


def verify_mask(mask, frame_shape, cfg, reasons=None):
    """按 opencv_close / opencv_new_close 的流程判定，返回 ((x,y,r,cov) 或 None, 掩膜)。

    reasons 传入一个 list 时，会把每个被拒候选的原因写进去（调试用）。
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
            if reasons is not None:
                reasons.append("面积 %d 不在 [%d, %d]"
                               % (area, area_min, area_max))
            continue

        (cx, cy), radius = cv2.minEnclosingCircle(cnt)
        if radius < r_min or radius > r_max:
            if reasons is not None:
                reasons.append("半径 %.0f 不在 [%.0f, %.0f]" % (radius, r_min, r_max))
            continue

        visible = disc_visibility(cx, cy, radius, fw, fh)
        if REQUIRE_FULL_CIRCLE and visible < CAND_EDGE_MIN_VISIBLE:
            if reasons is not None:
                reasons.append("圆有 %.0f%% 在画面外（多半是黑边/背景伪影）"
                               % (100 * (1.0 - visible)))
            continue

        if cfg["cov_on"] == "hull":
            ref_area = cv2.contourArea(cv2.convexHull(cnt))   # 忽略球一侧的缺口
        else:
            ref_area = area
        # 圆被画面边界切掉时，圆度按“画面内那部分圆”算：球进出画面很常见，
        # 拿整圆当分母的话贴在边上的球永远过不了。
        circle_area = math.pi * (radius ** 2) * max(visible, 1e-6)
        cov = ref_area / circle_area if circle_area > 0 else 0.0
        if cov <= cfg["cov_min"]:
            if reasons is not None:
                reasons.append("圆度 %.2f <= %.2f（r=%.0f）"
                               % (cov, cfg["cov_min"], radius))
            continue

        if best is None or cov > best[3]:
            best = (float(cx), float(cy), float(radius), float(cov))
    if best is None and reasons is not None and not reasons:
        reasons.append("形态学后没有轮廓（球太小或颜色断层）")
    return best, m


_last_reject_print = 0.0


def _report_reject(reasons):
    """把“为什么没找到”打到终端，最多每秒一次，避免刷屏。"""
    global _last_reject_print
    now = time.time()
    if now - _last_reject_print < 1.0:
        return
    _last_reject_print = now
    print("[未找到] " + " | ".join(reasons[:3]))


def mask_circles(mask, cfg, short):
    """在颜色掩膜上做霍夫圆。

    球被自己的黑条纹/logo 切成几块时，轮廓法的圆度会崩掉；但颜色掩膜的外边界
    仍然是个圆，霍夫直接在掩膜上就能把这个圆找出来。作为轮廓法找不到时的兜底。

    只在掩膜包围盒里跑：霍夫是整条流水线里最贵的一步（720p 上一次 8.7ms），
    而掩膜通常只占画面一小块，限制到包围盒后常降到 1ms 级别。
    """
    if not mask.any():
        return []
    x, y, bw, bh = mask_bbox(mask)
    if bw < 2 or bh < 2:
        return []
    pad = int(cfg["r_max_ratio"] * short) + 2
    x0, y0 = max(0, x - pad), max(0, y - pad)
    x1 = min(mask.shape[1], x + bw + pad)
    y1 = min(mask.shape[0], y + bh + pad)
    sub = mask[y0:y1, x0:x1]

    close_k = cfg["close_k"] | 1        # 保证是奇数
    clean = cv2.morphologyEx(sub, cv2.MORPH_CLOSE,
                             np.ones((close_k, close_k), np.uint8))
    open_k = cfg["open_k"] | 1
    clean = cv2.morphologyEx(clean, cv2.MORPH_OPEN,
                             np.ones((open_k, open_k), np.uint8))
    blur = cv2.GaussianBlur(clean, (9, 9), 2)
    circles = cv2.HoughCircles(
        blur, cv2.HOUGH_GRADIENT, dp=1.5,
        minDist=max(20, int(0.3 * short)), param1=100, param2=40,
        minRadius=int(cfg["r_min_ratio"] * short),
        maxRadius=int(cfg["r_max_ratio"] * short))
    if circles is None:
        return []
    return [(float(cx + x0), float(cy + y0), float(r)) for cx, cy, r in circles[0]]


def best_mask_circle(mask, cfg):
    """在掩膜上用霍夫找圆，返回圆内“球色占比”最高的若干个（按半径从大到小）。"""
    fh, fw = mask.shape[:2]
    short = min(fh, fw)
    ball = mask > 0
    found = []
    for x, y, r in mask_circles(mask, cfg, short):
        if not (0 <= x < fw and 0 <= y < fh):      # 圆心必须在画面内
            continue
        rr = int(round(r))
        x0, y0 = max(0, int(x) - rr), max(0, int(y) - rr)
        x1, y1 = min(fw, int(x) + rr + 1), min(fh, int(y) + rr + 1)
        if x1 <= x0 or y1 <= y0:
            continue
        disc = np.zeros((y1 - y0, x1 - x0), np.uint8)
        cv2.circle(disc, (int(x) - x0, int(y) - y0), rr, 255, -1)
        fill = float(ball[y0:y1, x0:x1][disc > 0].mean())
        if fill >= CIRCLE_MIN_FILL:
            found.append((x, y, r, fill))
    found.sort(key=lambda t: t[2], reverse=True)     # 球通常比脸/手大
    return found


def radial_radius(mask, x, y, max_r):
    """从 (x, y) 向四周射线量掩膜边界，取中位距离当半径。

    距离变换的峰值半径会被球皮黑纹/logo 挖出的洞坑住（球很圆但掩膜中间有空），
    这里沿 36 个方向量到底，取中位数：球被手/手臂粘住时只有少数方向会拖长，
    中位数不受影响。36 个方向（10° 一档）与 72 个的结果差不到 1 像素，
    但坐标只算一次、整批一起查表，比逐条射线便宜一个数量级。
    """
    h, w = mask.shape[:2]
    n_ray, n_t = 36, int(max_r)
    if n_t < 1:
        return 0.0
    ang = np.deg2rad(np.arange(n_ray) * (360.0 / n_ray))
    ts = np.arange(1, n_t + 1, dtype=np.float32)
    pxs = np.rint(x + np.cos(ang)[:, None] * ts[None, :]).astype(np.int32)
    pys = np.rint(y + np.sin(ang)[:, None] * ts[None, :]).astype(np.int32)
    inside = (pxs >= 0) & (pxs < w) & (pys >= 0) & (pys < h)
    np.clip(pxs, 0, w - 1, out=pxs)
    np.clip(pys, 0, h - 1, out=pys)
    hit = inside & (mask[pys, pxs] > 0)
    dists = []
    for row in hit:
        idx = np.nonzero(row)[0]
        if idx.size:
            dists.append(float(idx[-1] + 1))
    if not dists:
        return 0.0
    return float(np.median(dists))


def inscribed_circles(mask, min_r, max_r, limit=2):
    """在色掩膜上找“最大内切圆”，按半径从大到小返回。

    球被手/手臂连成一整块、或者被运动模糊拉长时，轮廓法和霍夫都拿它没办法；
    但球本身仍是这一块里最厚实的圆形区域 —— 距离变换的峰值就是球心，
    峰值大小就是内切半径。每找到一颗就把它从掩膜里抹掉，再找下一颗。

    只在掩膜的包围盒里做（距离变换是 O(宽×高)，全画幅一次 2~3ms，
    包围盒里通常只要 0.2ms）。
    """
    if not mask.any():
        return []
    bx, by, bw, bh = cv2.boundingRect(mask)
    if bw < 2 or bh < 2:
        return []
    work = mask[by:by + bh, bx:bx + bw]
    work = cv2.morphologyEx(work, cv2.MORPH_CLOSE,
                            np.ones((HEAL_K, HEAL_K), np.uint8))
    found = []
    for _ in range(limit):
        dist = cv2.distanceTransform(work, cv2.DIST_L2, 5)
        _minv, peak, _minloc, loc = cv2.minMaxLoc(dist)
        if not np.isfinite(peak) or peak < min_r or peak > max(bw, bh):
            break
        if peak <= max_r:      # 太厚的多半是墙/衣服，不是球：跳过但抹掉继续找
            r = max(float(peak), radial_radius(mask, loc[0] + bx, loc[1] + by, max_r))
            if r <= max_r:
                found.append((float(loc[0] + bx), float(loc[1] + by), r))
        cv2.circle(work, (int(loc[0]), int(loc[1])), max(1, int(peak * 1.15)), 0, -1)
    found.sort(key=lambda t: t[2], reverse=True)
    return found


_sticky = {"hsv": None}      # 上一帧命中的色模（H,S,V），用来避免在相近色模之间来回跳


def reset_sticky():
    _sticky["hsv"] = None


def _hyp_priority(hyp, ref):
    """和上一帧命中色的接近程度，越小越优先。"""
    h, s, _v = hyp
    dh = hue_distance(h, ref[0])
    ds = abs(s - ref[1])
    return dh * 2.0 + ds * 0.15


# ==========================================================
# ==== 径向边缘圆拟合：让圈真正“包住”球，而不是“塞进球” ====
# ==========================================================
# 内切圆（距离变换）只能取掩膜里放得下的最大圆：球被手/前臂连住、或被画面边缘
# 裁掉时，圆心会被拉走、半径会被“脖子”卡小。这里改成直接拟合**球的边界**：
# 沿射线找边缘点，只保留“梯度方向与半径方向平行”的那些 —— 球面上的边缘正好
# 满足这一点，而手、前臂、背景的边界是斜切/平行的，|cos| 很小，会被排除。
# 再对这些点做稳健最小二乘圆拟合，圆心和半径都不受连通块形状影响。
FIT_RAYS = 36               # 射线数（10° 一条；72 条只快 1 像素，代价翻倍）
FIT_SEARCH_LO = 0.55        # 射线搜索范围 = [0.55r, 1.75r]
FIT_SEARCH_HI = 1.75
FIT_ALIGN_MIN = 0.72        # 梯度方向与半径方向的最小 |cos|
FIT_GRAD_MIN = 24.0         # 边缘点梯度幅值下限（Sobel，8 位灰度上）
FIT_TRIM_FRAC = 0.16        # 残差超过 半径×此系数 的点视为离群点
FIT_MIN_POINTS = 6          # 至少这么多边界点才拟合（球被裁掉一半也够）
FIT_MIN_SUPPORT = 0.22      # 有效点数 / 射线数 的最低比例
FIT_SKIP_SUPPORT = 0.62     # 候选圆已经这么贴合边界时就不必再拟合（省时间）
FIT_MAX_ITER = 3            # 迭代次数（每轮按新圆重新打射线）
REFINE_TOP_K = 3            # 每帧只对半径最大的几个候选做边界拟合（省时间）
REFINE_TRY_MAX = 8          # 但最多考察这么多个候选（拟合失败的后面几个不再拟合）


def _radial_edge_points(gx, gy, x, y, r, min_r, max_r, n_rays=FIT_RAYS):
    """沿 n_rays 条射线找“边缘方向平行于半径”的点。

    每条射线可能穿过多个合格边缘（球皮黑条纹、手的轮廓、背景边界……），
    这里取**离当前半径最近**的那个，而不是梯度最强的那个：球的真实边界就在
    当前半径附近，取最强的会被手的轮廓带跑，圈越涨越大。
    """
    h, w = gx.shape
    lo = max(3.0, FIT_SEARCH_LO * r)
    hi = min(max_r * 1.05, FIT_SEARCH_HI * r)
    if hi <= lo:
        return []
    steps = int(math.ceil(hi - lo)) + 1
    if steps < 3:
        return []
    ang = np.arange(n_rays) * (2.0 * np.pi / n_rays)
    ux = np.cos(ang).astype(np.float32)
    uy = np.sin(ang).astype(np.float32)
    dists = lo + np.arange(steps, dtype=np.float32)
    px = np.rint(x + ux[:, None] * dists[None, :]).astype(np.int32)
    py = np.rint(y + uy[:, None] * dists[None, :]).astype(np.int32)
    inside = (px >= 1) & (px < w - 1) & (py >= 1) & (py < h - 1)
    xs = np.clip(px, 1, w - 2)
    ys = np.clip(py, 1, h - 2)
    gxs = np.where(inside, gx[ys, xs], 0.0)
    gys = np.where(inside, gy[ys, xs], 0.0)
    mag = np.hypot(gxs, gys)
    # 边缘方向（梯度法线）与半径方向的夹角：球的边界上两者平行
    cosd = np.abs(gxs * ux[:, None] + gys * uy[:, None]) / np.maximum(mag, 1e-6)
    peak = np.zeros_like(mag, dtype=bool)
    peak[:, 1:-1] = (mag[:, 1:-1] >= mag[:, :-2]) & (mag[:, 1:-1] >= mag[:, 2:])
    valid = peak & (mag >= FIT_GRAD_MIN) & (cosd >= FIT_ALIGN_MIN)
    cost = np.where(valid, np.abs(dists[None, :] - r), np.inf)
    j = np.argmin(cost, axis=1)
    rows = np.arange(n_rays)
    ok = np.isfinite(cost[rows, j])
    pts = []
    for k in np.nonzero(ok)[0]:
        jj = int(j[k])
        # 抛物线插值取亚像素位置
        m0, m1, m2 = mag[k, jj - 1], mag[k, jj], mag[k, jj + 1]
        den = float(m0 - 2.0 * m1 + m2)
        off = 0.5 * float(m0 - m2) / den if abs(den) > 1e-6 else 0.0
        off = max(-0.5, min(0.5, off))
        d = lo + jj + off
        pts.append((float(x + ux[k] * d), float(y + uy[k] * d)))
    return pts


def _fit_circle_ls(pts):
    """代数最小二乘圆拟合 x²+y² = A·x + B·y + C（对弧段也成立）。"""
    if len(pts) < 3:
        return None
    a = np.array(pts, np.float64)
    mat = np.column_stack([a[:, 0], a[:, 1], np.ones(len(a))])
    rhs = a[:, 0] ** 2 + a[:, 1] ** 2
    sol, _res, rank, _sv = np.linalg.lstsq(mat, rhs, rcond=None)
    if rank < 3:
        return None
    cx, cy = sol[0] / 2.0, sol[1] / 2.0
    rad2 = sol[2] + cx * cx + cy * cy
    if not np.isfinite(rad2) or rad2 <= 1.0:
        return None
    return float(cx), float(cy), float(math.sqrt(rad2))


def _disc_fill(mask, x, y, r):
    """圆内“球色”像素占比（圆超出画面时只统计画面内的部分）。"""
    h, w = mask.shape[:2]
    rr = int(max(2, round(r)))
    x0, y0 = max(0, int(round(x)) - rr), max(0, int(round(y)) - rr)
    x1, y1 = min(w, int(round(x)) + rr + 1), min(h, int(round(y)) + rr + 1)
    if x1 <= x0 or y1 <= y0:
        return 0.0
    disc = np.zeros((y1 - y0, x1 - x0), np.uint8)
    cv2.circle(disc, (int(round(x)) - x0, int(round(y)) - y0), rr, 255, -1)
    sel = disc > 0
    if not sel.any():
        return 0.0
    return float((mask[y0:y1, x0:x1][sel] > 0).mean())


def _support_of(pts, x, y, r, tol_frac=0.10):
    """这些点里有百分之多少真的落在圆上（按射线数归一化）。"""
    if not pts:
        return 0.0
    on = sum(1 for p in pts
             if abs(math.hypot(p[0] - x, p[1] - y) - r) <= tol_frac * r)
    return on / float(FIT_RAYS)


def refine_circle(grad, x, y, r, min_r, max_r, mask=None):
    """把候选圆修正成“贴合球边界”的圆，失败时返回 None。

    迭代：按当前圆心打射线取边界点 → 剔除离群点 → 最小二乘重拟合，
    圆心和半径一起收敛（球被画面裁掉一半时，弧段拟合照样能定出整圆）。

    三道闸门防止“拟合跑偏”：
      1. 半径限制在种子的 0.75~1.35 倍，圆心总位移 ≤ 0.7R；
      2. 拟合后的圆必须在边界上比原候选更贴合，否则退回原候选
         （这条保证“改了不会更差”）；
      3. 圆内还得是球色 —— 拟合很容易顺着人脸轮廓爬过去，用掩膜填充率挡住。
    """
    gx, gy = grad
    cx, cy, cr = float(x), float(y), float(r)
    seed_r = cr
    x0, y0 = cx, cy

    pts = _radial_edge_points(gx, gy, cx, cy, cr, min_r, max_r)
    seed_support = _support_of(pts, cx, cy, cr)
    if seed_support >= FIT_SKIP_SUPPORT:
        return None                      # 已经贴得很准，不再折腾（省时间）
    fit = None
    kept_arr = None
    for it in range(FIT_MAX_ITER):
        if it:
            pts = _radial_edge_points(gx, gy, cx, cy, cr, min_r, max_r)
        if len(pts) < FIT_MIN_POINTS:
            break
        f = _fit_circle_ls(pts)
        if f is None:
            break
        # 稳健剔除离群点（黑条纹、手、背景边缘残留）—— 用 numpy 向量算残差
        arr = np.asarray(pts, np.float64)
        for _ in range(2):
            fx, fy, fr = f
            res = np.abs(np.hypot(arr[:, 0] - fx, arr[:, 1] - fy) - fr)
            tol = max(2.5, FIT_TRIM_FRAC * fr)
            mask_ok = res <= tol
            if int(mask_ok.sum()) < FIT_MIN_POINTS:
                break
            arr = arr[mask_ok]
            f2 = _fit_circle_ls(arr)
            if f2 is None:
                break
            f = f2
        kept_arr = arr
        nx, ny, nr = f
        if nr < min_r or nr > max_r:
            break
        if not (0.75 * seed_r <= nr <= 1.35 * seed_r):
            break
        # 相对**原始种子**限制总位移：球和手粘连时，边界拟合很容易一步一步
        # 爬到手上/脸上（每轮都只动一点，累计却能挪一个半径）。
        if math.hypot(nx - x0, ny - y0) > max(6.0, 0.7 * seed_r):
            break
        cx, cy, cr = nx, ny, nr
        fit = (cx, cy, cr)
        pts = None            # 下一轮重新打射线
    if fit is None:
        return None
    # 闸门 2：拟合结果必须在边界上真的有更多支撑（用最后一轮保留下来的点估计）
    fit_support = (len(kept_arr) / float(FIT_RAYS)) if kept_arr is not None else 0.0
    if fit_support < max(seed_support + 0.02, FIT_MIN_SUPPORT):
        return None
    # 闸门 3：圆内还得是“球色”。拟合很容易顺着人脸的轮廓爬过去（梯度也对齐、
    # 支撑率还更高），但那个圆里有将近一半不是球色像素 —— 用掩膜填充率挡住。
    if mask is not None:
        seed_fill = _disc_fill(mask, x, y, r)
        fit_fill = _disc_fill(mask, fit[0], fit[1], fit[2])
        if fit_fill < max(0.60, seed_fill - 0.12):
            return None
    return fit


def _detect_once(hsv, gray, lab, shape, hyps, cfg, centers, reasons,
                 base_masks=None, stats=None, try_hough=True, grad=None):
    """一次完整的查找：轮廓法 + 最大内切圆 + 霍夫圆兜底，最后统一做边界拟合。

    三条路的候选**全部收集**起来，最后按半径挑最大的那个 —— 球是画面里最大的
    圆形色块。以前是“谁先过就返回谁”，结果脸（圆度更高）经常抢在球前面。

    base_masks 是本帧已经算好的 (色相窗口 ∩ 色度下限) 掩膜，和 hyps 一一对应。
    两遍检测共用它：第二遍只需要再按 a/b 归属切开（split_by_owner），
    不用把色相/色度那两步（每次 5~6ms）重算一遍。

    每个候选在入库前都会用 refine_circle 重新拟合球边界：内切圆被“脖子”卡小、
    圆心被连成一块的手拉偏时，这一步能把它拉回真正的球。
    """
    fh, fw = shape[:2]
    short = min(fh, fw)
    raw_view = np.zeros((fh, fw), np.uint8)
    keep_view = np.zeros((fh, fw), np.uint8)
    min_r = cfg["r_min_ratio"] * short
    max_r = cfg["r_max_ratio"] * short
    cands = []      # (半径, x, y, 色相, 饱和度, 色模, 原始掩膜, 判定后掩膜, 圆度)

    if grad is None:
        grad = (cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3),
                cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3))

    def accept(x, y, radius):
        """外观确认：色度够浓，或者比周围背景明显更红（弱光/过曝都靠后者）。"""
        ok, why = ball_look_ok(hsv, gray, lab, x, y, radius, stats)
        if not ok and reasons is not None:
            reasons.append("圆内%s" % why)
        return ok

    def keep(x, y, radius, hyp, raw, judged, cov):
        s_med, h_mean = circle_stats(hsv, x, y, radius)
        cands.append((float(radius), int(x), int(y), h_mean, s_med, hyp,
                      raw, judged, float(cov)))

    def consider(x, y, radius, hyp, raw, judged, cov, strict_fill=False):
        """先把候选收进来（不做拟合），等全部收完再挑前几个做边界拟合。

        拟合一次要打 1~4 轮射线（约 1~2ms），而一帧最多能有十几个候选；
        只对“半径最大”的那几个做就够了 —— 球本来就是画面里最大的圆形色块。
        strict_fill=True 表示这个候选只经过了宽松的几何检查（内切圆/霍夫），
        必须额外确认“圆内确实是球色”。
        """
        cands.append((float(radius), int(x), int(y), 0.0, 0.0, hyp,
                      raw, judged, float(cov), bool(strict_fill)))

    def refine_and_accept(c, do_fit=True):
        """对单个候选做边界拟合 + 填充率检查 + 外观确认。"""
        radius, x, y, _h, _s, hyp, raw, judged, cov, strict_fill = c
        need_fill = strict_fill
        if do_fit:
            fit = refine_circle(grad, x, y, radius, min_r, max_r, mask=raw)
            if fit is not None:
                x, y, radius = fit
                need_fill = True      # 圆被改动过，重新确认一次
        # 圆里必须是球色：拟合很容易把人脸轮廓当成球边界，圆内填充率会掉到
        # 0.5 左右（实测 0.54），却因为半径最大被选中。
        if need_fill and _disc_fill(raw, x, y, radius) < CAND_MIN_FILL:
            return None
        if not accept(x, y, radius):
            return None
        s_med, h_mean = circle_stats(hsv, x, y, radius)
        return (float(radius), int(x), int(y), h_mean, s_med, hyp, raw, judged, cov)

    # 上一帧命中的色模优先：否则每帧可能在几个相近的色模之间来回跳，圈一直抖
    # （排序要在算 base_masks 之前做，因为调用方按排序后的下标缓存掩膜）
    if _sticky["hsv"] is not None:
        ref = _sticky["hsv"]
        hyps = sorted(hyps, key=lambda t: _hyp_priority(t[1], ref))

    # 色度刻度按当前帧缩放：掩膜里只留“够浓”的像素，弱光下 S 门槛放进来一堆
    # 肤色，球和脸/手会连成一块，面积超过上限直接被否。
    stats = stats or scene_stats(lab)
    floor = stats["floor"]

    def raw_mask_of(idx):
        if base_masks is not None:
            raw = base_masks[idx]
        else:
            raw = hypothesis_mask(hsv, hyps[idx][1])
            if floor > 1.0:
                raw = chroma_gate(lab, raw, floor)
        if centers is not None:
            raw = split_by_owner(raw, lab, centers, hyps[idx][2])
        return raw

    # 1) 轮廓法：形状门槛 + 圆度，最严格的一路
    for i, (_score, hyp, mode_idx) in enumerate(hyps):
        raw = raw_mask_of(i)
        # 先做一次闭运算把球皮黑条纹造成的小裂缝补上，再判“是不是非球色块”。
        # 顺序反了的话，被条纹切成的楔形块会被“太细长”规则整块删掉，球只剩一个角。
        # 这里用独立核（HEAL_K），不改各判定标准自己的 5x5 / 11x11。
        k = max(cfg["close_k"] | 1, HEAL_K)
        healed = cv2.morphologyEx(raw, cv2.MORPH_CLOSE,
                                  np.ones((k, k), np.uint8))
        mask = drop_non_ball_blobs(healed, short, lab, hsv, stats["blob_min"])
        if healed.any() and not mask.any() and reasons is not None:
            reasons.append("色块被形状门槛删掉（太细长 / 大而不圆）")
        found, judged = verify_mask(mask, shape, cfg, reasons)
        if not raw_view.any():
            raw_view, keep_view = raw, judged
        if found is not None:
            x, y, radius, cov = found
            consider(x, y, radius, hyp, raw, judged, cov)

    # 2) 最大内切圆（距离变换）：球被手/手臂连成一整块、或被运动模糊拉长时，
    #    轮廓和霍夫都崩，但球仍是那块里最厚实的圆。
    #    只有轮廓法一条候选都没有时才跑 —— 能过轮廓法的帧根本不需要它，
    #    而每帧无条件跑要白花 4~5ms。
    if not cands:
        for i, (_score, hyp, mode_idx) in enumerate(hyps):
            raw = raw_mask_of(i)
            for x, y, radius in inscribed_circles(raw, min_r, max_r):
                consider(x, y, radius, hyp, raw, raw, 0.0, strict_fill=True)

    # 3) 霍夫圆：救回被黑条纹切碎的球。比较贵，前面一条候选都没有时才跑。
    if not cands and try_hough:
        for i, (_score, hyp, mode_idx) in enumerate(hyps):
            raw = raw_mask_of(i)
            for x, y, radius, _fill in best_mask_circle(raw, cfg):
                consider(x, y, radius, hyp, raw, raw, 0.0, strict_fill=True)

    if cands:
        # 只对半径最大的几个候选做边界拟合（球是画面里最大的圆形色块）；
        # 其余的仍然参与“填充率 + 外观”筛选，只是不做拟合，避免把候选池缩得太小。
        cands.sort(key=lambda t: (t[0], t[8]), reverse=True)
        picked = None
        for k, c in enumerate(cands[:REFINE_TRY_MAX]):
            got = refine_and_accept(c, do_fit=(k < REFINE_TOP_K))
            if got is not None and (picked is None or got[0] > picked[0]):
                picked = got
        if picked is not None:
            radius, x, y, h_mean, s_med, hyp, raw, judged, _cov = picked
            _sticky["hsv"] = hyp
            return (x, y, int(radius), h_mean, s_med), raw, judged, []
    return None, raw_view, keep_view, reasons


def auto_detect(frame, hyps, centers=None, mode=VERIFY_MODE):
    """逐帧查找，返回 (检测结果或 None, 原始色掩膜, 通过形状门槛的掩膜)。

    两遍：
      1. 整块颜色掩膜（和以前一样）——绝大多数情况在这里就找到了；
      2. 只有第 1 遍失败时才做：按 a/b 归属把掩膜切开，让球从“和脸/手连成
         一块的大色块”里分离出来。这一遍额外要求分割出的圆内**色度明显高于
         周围背景**（球比背景红得多，皮肤没有），否则孤立的脸/手会被切成
         小圆块误检。

    两遍共用同一份 base_masks（色相窗口 ∩ 色度下限）：这一步每帧要 5~6ms，
    重算一次纯属浪费，第二遍真正多出来的只有按归属切开。
    """
    if not hyps:
        return None, np.zeros(frame.shape[:2], np.uint8), np.zeros(frame.shape[:2], np.uint8)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cfg = VERIFY_MODES[mode]
    stats = scene_stats(lab)
    grad = (cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3),
            cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3))

    base_masks = []
    chroma_ok = None
    if stats["floor"] > 1.0:
        chroma_ok = chroma_keep_mask(lab, stats["floor"])
    for _score, hyp, _idx in hyps:
        m = hypothesis_mask(hsv, hyp)
        if chroma_ok is not None:
            m = cv2.bitwise_and(m, chroma_ok)
        base_masks.append(m)

    reasons = []
    best, raw_view, keep_view, _ = _detect_once(
        hsv, gray, lab, frame.shape, hyps, cfg, None, reasons,
        base_masks=base_masks, stats=stats, grad=grad)
    if best is None and centers is not None and len(centers) > 1:
        split_reasons = []
        best2, raw2, keep2, _ = _detect_once(
            hsv, gray, lab, frame.shape, hyps, cfg, centers, split_reasons,
            base_masks=base_masks, stats=stats, try_hough=False, grad=grad)
        if best2 is not None:
            best, raw_view, keep_view = best2, raw2, keep2
            reasons = []
        else:
            raw_view, keep_view = raw2 if raw2.any() else raw_view, keep2
            if split_reasons:
                reasons = split_reasons

    if best is None and DEBUG_REJECT:
        if reasons:
            _report_reject(reasons)
        else:
            _report_reject(["画面里没有可用的颜色假设（没有足够鲜艳/够大的色块）"])
    return best, raw_view, keep_view


# ==========================================================
# ==================== 显示 / 入口 =========================
# ==========================================================
def draw(frame, det, hyps, mode, status, status_color):
    """主窗口标注：圈、状态、当前色模列表（掩膜另开窗口显示）。"""
    fh, fw = frame.shape[:2]
    if det is not None:
        x, y, radius = (int(round(v)) for v in det[:3])
        radius = max(3, radius)
        cv2.circle(frame, (x, y), radius, (0, 255, 0), 3)
        cv2.circle(frame, (x, y), 4, (0, 0, 255), -1)

    cv2.putText(frame, status, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)
    cv2.putText(frame, "mean shift | verify: %s" % mode, (20, fh - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

    for i, (_score, (hh, ss, vv), _idx) in enumerate(hyps):
        cv2.putText(frame, "mode%d: H%d S%d V%d" % (i, int(hh), int(ss), int(vv)),
                    (20, 70 + 20 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    return frame


def draw_timing(frame, fps, detect_ms, total_ms):
    """右上角显示实测帧率/耗时，方便判断卡顿是算法还是采集造成的。"""
    text = "%.1f fps | detect %.0fms | frame %.0fms" % (fps, detect_ms, total_ms)
    (tw, _th), _base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
    cv2.putText(frame, text, (max(4, frame.shape[1] - tw - 8), 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
    return frame


def open_capture(source):
    if isinstance(source, str) and source.isdigit():
        source = int(source)
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise SystemExit("打不开视频源: %s（摄像头被占用就换索引 1/2，或直接传视频路径）"
                         % (source,))
    # 关键：把驱动的帧队列压到 1。默认摄像头驱动会攒几十帧，处理慢于采集时
    # 队列越排越长，画面就越来越“滞后于现实”（这也是“延迟很高”的主因之一）。
    # 之后每帧先 grab 掉积压，只处理最新的一帧。
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if isinstance(source, int):
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if (w, h) != (640, 480):
            print("注意：摄像头实际输出 %dx%d（请求 640x480 被驱动忽略了）。"
                  "窗口会按实际比例显示，不会留黑边。" % (w, h))
    return cap


def read_latest(cap):
    """取“最新”的一帧：先把积压的旧帧丢掉，再读最后一帧。

    排队等待采集只会让延迟线性增长；宁可丢帧也要让画面对得上现在。
    返回 (ok, frame)。
    """
    pending = int(cap.get(cv2.CAP_PROP_BUFFERSIZE) or 0)
    for _ in range(max(0, pending - 1)):
        if not cap.grab():
            break
    return cap.read()


def parse_args(argv):
    """返回 (视频源, 判定标准)，支持 `meanshift_auto.py <src> --verify new_close`。"""
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
        print("未知判定标准 %r，可选 %s，改用 %s" % (mode, VERIFY_ORDER, VERIFY_MODE))
        mode = VERIFY_MODE
    if rest:
        src = rest[0]
    return src, mode


def save_capture(raw_frame, annotated, mask):
    os.makedirs(CAPTURE_DIR, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    paths = []
    for name, img in ((stamp + "_raw.png", raw_frame),
                      (stamp + "_annotated.png", annotated),
                      (stamp + "_mask.png", mask)):
        path = os.path.join(CAPTURE_DIR, name)
        cv2.imwrite(path, img)
        paths.append(path)
    return paths


def main():
    source, mode = parse_args(sys.argv)
    cap = open_capture(source)

    window = "Mean Shift Basketball Detection"
    mask_window = "Mask"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window, 960, 720)
    cv2.namedWindow(mask_window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(mask_window, 640, 480)
    print("mean shift 逐帧找球：每帧独立检测，不用上一帧的结果。"
          "v=切换判定标准  s=存图  q=退出")
    print("判定标准: %s" % mode)

    hyps, mode_centers = [], None
    shown = None            # 平滑后的显示位置 (x, y, r)
    vel = (0.0, 0.0)        # 显示位置的估计速度（像素/帧），用于预测
    frame_count = 0
    t_prev = time.time()
    fps_avg = 0.0
    x0 = y0 = 0
    x1 = y1 = None
    cropped = False

    while True:
        ok, frame = read_latest(cap)
        if not ok:
            break
        raw_frame = frame.copy()
        frame_count += 1

        # 实测帧率：既用来显示，也用来决定要不要跳过重聚类（见下）
        now = time.time()
        dt = now - t_prev
        t_prev = now
        if dt > 1e-6:
            fps_avg = 1000.0 if fps_avg == 0 else 0.9 * fps_avg + 0.1 * (1.0 / dt)

        if frame_count == 1:
            fh0, fw0 = frame.shape[:2]
            cv2.resizeWindow(window, max(320, fw0 * 3 // 4), max(240, fh0 * 3 // 4))
            cv2.resizeWindow(mask_window, max(240, fw0 * 3 // 5), max(180, fh0 * 3 // 5))

        # 裁掉相机填充的黑边，只在有内容的区域里找球。
        # 这一步要对整幅画面做分位数统计（720p 上约 6ms），但画面构图基本不变，
        # 所以每 N 帧才重算一次，其余帧沿用上次的结果。
        if frame_count % CONTENT_EVERY == 1:
            x0, y0, x1, y1 = content_box(frame)
            cropped = (x1 - x0, y1 - y0) != (frame.shape[1], frame.shape[0])
            if frame_count == 1 and cropped:
                print("检测区域: x[%d,%d) y[%d,%d)（已裁掉相机黑边，原画面 %dx%d）"
                      % (x0, x1, y0, y1, frame.shape[1], frame.shape[0]))
        work_full = frame[y0:y1, x0:x1] if cropped else frame
        dx, dy = (x0, y0) if cropped else (0, 0)

        # 检测分辨率上限：摄像头若给的是 1280x720，直接在原尺寸上算要 40ms/帧。
        # 统一缩到 PROCESS_WIDTH 宽再检测（圆的尺寸门槛都是相对短边的比例，
        # 缩放不影响判据），坐标最后再换算回去。
        scale = 1.0
        if work_full.shape[1] > PROCESS_WIDTH:
            scale = PROCESS_WIDTH / float(work_full.shape[1])
            work = cv2.resize(
                work_full,
                (PROCESS_WIDTH, int(round(work_full.shape[0] * scale))),
                interpolation=cv2.INTER_AREA)
        else:
            work = work_full

        # 聚类每个像素的归属不变，但每帧重算色模太贵；这里固定间隔重算。
        # 固定了 KMEANS_SEED，所以同样的画面每次得到同样的色模，不会跳。
        # 帧率吃紧时（<25fps）自动把间隔放大一倍：色模本来就很稳，
        # 少算几次几乎不影响结果，却能明显减少卡顿。
        every = CLUSTER_EVERY if fps_avg <= 0 or fps_avg >= 25 else CLUSTER_EVERY * 2
        if frame_count % every == 1 or not hyps:
            small = cv2.resize(work, (SMALL_W, SMALL_H), interpolation=cv2.INTER_AREA)
            hyps, mode_centers = color_hypotheses(small)

        # 每帧都完整找一次球，结果只属于这一帧
        det, raw_mask, keep_mask = auto_detect(work, hyps, mode_centers, mode)
        t_det = time.time()
        found = det is not None
        if found:
            inv = 1.0 / scale
            det = (det[0] * inv + dx, det[1] * inv + dy, det[2] * inv)  # 换回原尺寸坐标
            # 显示平滑：**先按速度预测，再按误差大小自适应修正**。
            # 原来固定权重 0.45 相当于一个 τ≈1.7 帧的低通，球一快圈就明显落在后面
            # （录屏里能看到圈停在球上一个位置）；改成“预测 + 自适应”后：
            #   球基本不动 → 预测≈原值，权重低，仍然压抖；
            #   球快速移动 → 预测先跟上，且误差大时权重抬到 1，等于不平滑，零滞后。
            if shown is None:
                shown = tuple(float(v) for v in det[:3])
                vel = (0.0, 0.0)
            else:
                px = shown[0] + vel[0]
                py = shown[1] + vel[1]
                gap = math.hypot(det[0] - px, det[1] - py)
                alpha = SMOOTH_POS + (1.0 - SMOOTH_POS) * min(
                    1.0, gap / max(4.0, SMOOTH_GAP_FRAC * max(4.0, det[2])))
                nx = px + alpha * (det[0] - px)
                ny = py + alpha * (det[1] - py)
                rad_gap = abs(det[2] - shown[2])
                r_alpha = SMOOTH_R + (1.0 - SMOOTH_R) * min(
                    1.0, rad_gap / max(2.0, 0.25 * max(4.0, det[2])))
                nz = shown[2] + r_alpha * (det[2] - shown[2])
                vel = (VEL_ALPHA * vel[0] + (1.0 - VEL_ALPHA) * (nx - shown[0]),
                       VEL_ALPHA * vel[1] + (1.0 - VEL_ALPHA) * (ny - shown[1]))
                shown = (nx, ny, nz)
        else:
            shown = None
            vel = (0.0, 0.0)

        status, s_color = (("Ball Detected", (0, 255, 0)) if found
                           else ("Searching...", (0, 0, 255)))
        frame = draw(frame, shown, hyps, mode, status, s_color)
        frame = draw_timing(frame, fps_avg,
                            (t_det - now) * 1000.0,
                            (time.time() - now) * 1000.0)

        # 掩膜窗口：白 = 通过形状门槛的颜色区，红 = 被判为"不是球"而删掉的颜色区
        # 掩膜是在缩小后的 work 上算的，先按最近邻放大回检测区域，再贴回整幅画面，
        # 这样两个窗口仍然像素对齐。
        if scale != 1.0:
            ms = (work_full.shape[1], work_full.shape[0])
            raw_mask = cv2.resize(raw_mask, ms, interpolation=cv2.INTER_NEAREST)
            keep_mask = cv2.resize(keep_mask, ms, interpolation=cv2.INTER_NEAREST)
        raw_canvas = align_mask(raw_mask, frame.shape, (x0, y0, x1, y1), cropped)
        keep_canvas = align_mask(keep_mask, frame.shape, (x0, y0, x1, y1), cropped)
        mask_view = cv2.cvtColor(raw_canvas, cv2.COLOR_GRAY2BGR)
        mask_view[(raw_canvas > 0) & (keep_canvas == 0)] = (0, 0, 255)
        if shown is not None:
            cv2.circle(mask_view, (int(shown[0]), int(shown[1])),
                       max(3, int(shown[2])), (0, 255, 0), 1)
        cv2.putText(mask_view, "MASK %s | white=kept  red=rejected"
                    % ("found" if found else "none"),
                    (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)

        cv2.imshow(window, frame)
        cv2.imshow(mask_window, mask_view)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):
            break
        if key == ord('v'):
            mode = VERIFY_ORDER[(VERIFY_ORDER.index(mode) + 1) % len(VERIFY_ORDER)]
            print("判定标准切换为: %s" % mode)
        if key == ord('s'):
            print("已保存:", " / ".join(save_capture(raw_frame, frame, raw_canvas)))

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()


