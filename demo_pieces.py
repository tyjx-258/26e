# demo_pieces.py - 碎片识别 Demo（降分辨率 + 局部裁剪 + 凸包去缺口）
#
# 用途：验证碎片能否稳定分割、能否拿到正确多边形顶点。屏幕上用不同颜色画出每片的
#       轮廓、朝向箭头、质心十字和序号；控制台每秒打印帧率、耗时与几何结果。
# 运行：MaixVision 打开本工程文件夹 -> 单独运行本文件（与 main.py 独立）。
# 配色：深色底板 + 白色碎片；若改成浅底深片，把 PIECE_INVERT 置 True 并改阈值。
# 路线：缩小图 -> binary -> find_blobs -> 裁剪小块 -> 局部取连通区 -> 逐行扫描 -> 凸包 -> 简化
#
# 三条硬约束：merge 必须为 False（合并包围盒相交的色块，斜放会并肩）；不做逐像素
# 生长的连通区隔离（会掉到 1 帧/秒）；轮廓用 draw_line 画（draw_edges 要求 >=4 顶点）。
# MaixPy 没有通用轮廓/多边形拟合接口，也没有可用的 ORB（官方标注 TODO），自行实现。

import math
import time

import numpy as np
from maix import app, camera, display, image

# ---- 相机 ----
CAM_W = 640
CAM_H = 480
EXPOSURE_US = 26000          # 现场实测值（侧光偏暗，比默认 16000 长）
GAIN_VALUE = 500

# ---- 算力 ----
WORK_SCALE = 2               # 识别在 1/2 分辨率上进行：像素数降到 1/4，坐标最后乘回来

# ---- 分割（阈值与像素数按原分辨率描述，代码里按 WORK_SCALE 折算）----
PIECE_INVERT = False         # True：目标比背景暗（浅底 + 深色碎片）
PIECE_THRESHOLDS = [[20, 100, -30, 30, -30, 30]]       # LAB：白片放宽亮度，收紧色度
MIN_PIXELS = 800             # 碎片最小像素数
MAX_PIXELS = 80000           # 碎片最大像素数：挡掉把底板/背景当成一片的情况
BBOX_PAD = 3                 # 裁剪时包围盒外扩像素（工作分辨率）
CLOSE_SIZE = 0               # >0 时对掩膜做闭运算，填补碎片内部空洞（牌面图案用）
OPEN_SIZE = 0                # 开运算会削掉尖角，白片场景先关闭

PATCH_THRESHOLDS = [[50, 100, -30, 30, -30, 30]]       # 二值块上再取连通区的阈值

# ---- 轮廓（阈值单位是工作分辨率像素）----
EPSILON_RATIO = 0.01         # 简化阈值 = 轮廓周长 × 该比例
MIN_EPSILON = 2.0            # 下限不能太小：斜放矩形在角上会多出 2px 的假顶点
MAX_EPSILON = 4.0
MIN_CONTOUR_POINTS = 8
MERGE_DISTANCE = 8.0         # 相邻顶点近于此距离就合并成一个（栅格化会把一个角裂成两点）
COMPENSATE_PX = 4.0     # 轮廓沿质心方向外扩的像素数，补偿阈值造成的内缩

# ---- 显示 ----
PIECE_COLORS = [
    (image.COLOR_RED, "red"),
    (image.COLOR_GREEN, "green"),
    (image.COLOR_BLUE, "blue"),
    (image.COLOR_YELLOW, "yellow"),
    (image.COLOR_PURPLE, "purple"),
    (image.COLOR_ORANGE, "orange"),
]

# ---- 节奏 ----
ANALYZE_INTERVAL_SEC = 0.5   # 识别间隔：越小越实时、越费算力
PRINT_INTERVAL_SEC = 1.0

WORK_W = CAM_W // WORK_SCALE
WORK_H = CAM_H // WORK_SCALE


def setup_camera(cam):
    """按固定值设置曝光与增益；失败只提示，让预览继续可用。"""
    try:
        cam.exposure(EXPOSURE_US)
        cam.gain(GAIN_VALUE)
    except Exception as e:
        print("set exposure/gain failed:", e)


def build_work_image(img):
    """按 WORK_SCALE 缩小图像；识别全部在小图上做，省算力。"""
    if WORK_SCALE <= 1:
        return img
    return img.resize(WORK_W, WORK_H)


def work_pixel_limits():
    """返回工作分辨率下的 (最小像素数, 最大像素数)。"""
    scale_area = WORK_SCALE * WORK_SCALE
    min_pixels = max(20, MIN_PIXELS // scale_area)
    max_pixels = max(min_pixels, MAX_PIXELS // scale_area)
    return min_pixels, max_pixels


def scan_contour(mask):
    """逐行扫描取轮廓：碎片是凸的，每行前景连续，所以左右极值即完整轮廓，
    一次 numpy argmax 算完整块（非凸碎片需换成沿边界追踪）。"""
    rows = np.flatnonzero(mask.any(axis=1))
    if rows.size == 0:
        return []
    band = mask[rows]
    left = np.argmax(band, axis=1)
    right = band.shape[1] - 1 - np.argmax(band[:, ::-1], axis=1)
    points = [(int(x), int(y)) for x, y in zip(left, rows)]
    points += [(int(x), int(y)) for x, y in zip(right[::-1], rows[::-1])]
    return points


def contour_perimeter(points):
    """闭合轮廓的周长（像素）。"""
    total = 0.0
    count = len(points)
    for i in range(count):
        x1, y1 = points[i]
        x2, y2 = points[(i + 1) % count]
        total += math.hypot(x2 - x1, y2 - y1)
    return total


def point_line_distance(point, a, b):
    """点到直线 ab 的距离，供 Douglas-Peucker 使用。"""
    px, py = point
    ax, ay = a
    bx, by = b
    dx, dy = bx - ax, by - ay
    if dx == 0.0 and dy == 0.0:
        return math.hypot(px - ax, py - ay)
    return abs(dy * px - dx * py + bx * ay - by * ax) / math.hypot(dx, dy)


def simplify(points, epsilon):
    """Douglas-Peucker：把折线压缩成关键顶点，首尾保留。"""
    if len(points) < 3:
        return list(points)
    first, last = points[0], points[-1]
    index, max_distance = 0, 0.0
    for i in range(1, len(points) - 1):
        distance = point_line_distance(points[i], first, last)
        if distance > max_distance:
            index, max_distance = i, distance
    if max_distance <= epsilon:
        return [first, last]
    head = simplify(points[:index + 1], epsilon)
    tail = simplify(points[index:], epsilon)
    return head[:-1] + tail


def simplify_closed(points, epsilon):
    """闭合轮廓的简化：先在最远点拆成两段再分别简化（首尾重合会让 Douglas-Peucker 失效）。"""
    if len(points) < 3:
        return list(points)
    x0, y0 = points[0]
    far = max(range(1, len(points)),
              key=lambda i: (points[i][0] - x0) ** 2 + (points[i][1] - y0) ** 2)
    head = simplify(points[:far + 1], epsilon)
    tail = simplify(points[far:] + [points[0]], epsilon)
    return head[:-1] + tail[:-1]


def merge_close_vertices(points, min_distance):
    """合并相距过近的顶点：栅格化会让一个真实拐角裂成相距几像素的两点。
    碎片每条边不小于 2cm（工作分辨率约 24 像素），故 8 像素内必是同一角的重复。"""
    merged = []
    for point in points:
        if merged and math.hypot(point[0] - merged[-1][0], point[1] - merged[-1][1]) < min_distance:
            last = merged[-1]
            merged[-1] = ((last[0] + point[0]) / 2.0, (last[1] + point[1]) / 2.0)
        else:
            merged.append(point)

    if len(merged) >= 3:
        first, last = merged[0], merged[-1]
        if math.hypot(first[0] - last[0], first[1] - last[1]) < min_distance:
            center = ((first[0] + last[0]) / 2.0, (first[1] + last[1]) / 2.0)
            merged = [center] + merged[1:-1]
    return merged


def convex_hull(points):
    """凸包（Andrew 单调链）：碎片一定是凸多边形，凸包能一次抹掉图案缺口与边界锯齿。"""
    unique = sorted(set(points))
    if len(unique) < 3:
        return unique

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower = []
    for point in unique:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0:
            lower.pop()
        lower.append(point)

    upper = []
    for point in reversed(unique):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0:
            upper.pop()
        upper.append(point)

    return lower[:-1] + upper[:-1]


def polygon_centroid(points):
    """按顶点序列计算多边形面积质心，返回 ((cx, cy), 像素面积)。"""
    area2 = 0.0
    sum_x = 0.0
    sum_y = 0.0
    count = len(points)
    for i in range(count):
        x1, y1 = points[i]
        x2, y2 = points[(i + 1) % count]
        cross = x1 * y2 - x2 * y1
        area2 += cross
        sum_x += (x1 + x2) * cross
        sum_y += (y1 + y2) * cross
    if abs(area2) < 1e-6:
        return points[0], 0.0
    return (sum_x / (3.0 * area2), sum_y / (3.0 * area2)), abs(area2) / 2.0


def expand_polygon(points, distance):
    """把顶点沿质心方向外扩 distance 像素，补偿阈值只保留亮边造成的内缩。

    属近似均匀外扩（对角点略多、对长边中点略少），用于抵消 1~3 像素的系统偏差够用。
    """
    count = len(points)
    if count < 3 or distance <= 0:
        return list(points)
    cx = sum(p[0] for p in points) / count
    cy = sum(p[1] for p in points) / count
    expanded = []
    for x, y in points:
        radius = math.hypot(x - cx, y - cy)
        if radius < 1e-6:
            expanded.append((x, y))
        else:
            expanded.append((x + (x - cx) / radius * distance,
                             y + (y - cy) / radius * distance))
    return expanded


def longest_edge(points):
    """返回最长边的 (角度, 长度, 端点1, 端点2)；角度以最长边为参考并归一化到 0~180 度。"""
    best = (0.0, -1.0, points[0], points[0])
    count = len(points)
    for i in range(count):
        x1, y1 = points[i]
        x2, y2 = points[(i + 1) % count]
        length = math.hypot(x2 - x1, y2 - y1)
        if length > best[1]:
            angle = math.degrees(math.atan2(y2 - y1, x2 - x1)) % 180.0
            best = (angle, length, (x1, y1), (x2, y2))
    return best


def pick_patch_component(patch, seed_x, seed_y, min_pixels):
    """在局部二值块上再取一次连通区，剔掉框进来的邻片，只留离质心最近的那块。"""
    blobs = patch.find_blobs(PATCH_THRESHOLDS, pixels_threshold=min_pixels, merge=False)
    best = None
    best_distance = 0.0
    for blob in blobs:
        dx = blob[0] + blob[2] / 2.0 - seed_x
        dy = blob[1] + blob[3] / 2.0 - seed_y
        distance = dx * dx + dy * dy
        if best is None or distance < best_distance:
            best, best_distance = blob, distance
    return best


def find_piece_shapes(img):
    """找出所有碎片，返回 [{"verts", "center", "angle", "area"}, ...]（坐标已换算回原图）。"""
    work = build_work_image(img)
    mask_img = work.binary(PIECE_THRESHOLDS, invert=PIECE_INVERT, copy=True)
    if CLOSE_SIZE > 0:      # 闭运算：补上碎片内部的图案空洞
        mask_img = mask_img.dilate(CLOSE_SIZE).erode(CLOSE_SIZE)
    if OPEN_SIZE > 0:       # 开运算：切断碎片之间只有几像素宽的细连接，避免并片
        mask_img = mask_img.erode(OPEN_SIZE).dilate(OPEN_SIZE)

    min_pixels, max_pixels = work_pixel_limits()
    # merge 必须为 False：官方文档写明它合并的是"包围盒相交"的色块，
    # 碎片斜放时包围盒互相压到，一两厘米间隙也会被并成一片。
    blobs = work.find_blobs(PIECE_THRESHOLDS, invert=PIECE_INVERT,
                            pixels_threshold=min_pixels, merge=False)

    shapes = []
    for blob in blobs:
        pixels = blob.pixels()
        if pixels > max_pixels:
            continue        # 太大，多半是把底板或背景当成了一片

        x, y, w, h = blob[0], blob[1], blob[2], blob[3]
        x0 = max(0, x - BBOX_PAD)
        y0 = max(0, y - BBOX_PAD)
        x1 = min(WORK_W, x + w + BBOX_PAD)
        y1 = min(WORK_H, y + h + BBOX_PAD)
        if x1 - x0 < 5 or y1 - y0 < 5:
            continue

        patch = mask_img.crop(x0, y0, x1 - x0, y1 - y0)
        sub = pick_patch_component(patch, blob.cx() - x0, blob.cy() - y0, min_pixels)
        if sub is None:
            continue
        sub_x, sub_y, sub_w, sub_h = sub[0], sub[1], sub[2], sub[3]
        if sub_w < 5 or sub_h < 5:
            continue
        patch = patch.crop(sub_x, sub_y, sub_w, sub_h)
        base_x, base_y = x0 + sub_x, y0 + sub_y

        local = image.image2cv(patch)[:, :, 0] > 128
        contour = scan_contour(local)
        if len(contour) < MIN_CONTOUR_POINTS:
            continue

        absolute = [(px + base_x, py + base_y) for px, py in contour]
        hull = convex_hull(absolute)        # 先抹掉牌面图案缺口与边界锯齿
        if len(hull) < 3:
            continue

        epsilon = min(max(contour_perimeter(hull) * EPSILON_RATIO, MIN_EPSILON), MAX_EPSILON)
        verts_small = merge_close_vertices(simplify_closed(hull, epsilon), MERGE_DISTANCE)
        if len(verts_small) < 3:
            continue

        verts = [(px * WORK_SCALE, py * WORK_SCALE) for px, py in verts_small]
        verts = expand_polygon(verts, COMPENSATE_PX)   # 补偿阈值只保留亮边造成的内缩
        center, area = polygon_centroid(verts)
        angle, _, _, _ = longest_edge(verts)
        shapes.append({"verts": verts, "center": center, "angle": angle, "area": area})
    return shapes


def color_of(index):
    """返回第 index 片碎片使用的颜色与颜色名，供画图与日志共用。"""
    return PIECE_COLORS[index % len(PIECE_COLORS)]


def draw_shape(img, shape, index):
    """按碎片序号取色，画出轮廓、朝向箭头、质心十字和序号标注。"""
    color, _ = color_of(index)
    # 逐条边画：draw_edges() 要求至少 4 个顶点，三角形碎片会直接抛错
    verts = shape["verts"]
    count = len(verts)
    for i in range(count):
        x1, y1 = verts[i]
        x2, y2 = verts[(i + 1) % count]
        img.draw_line(int(x1), int(y1), int(x2), int(y2), color)

    _, _, start, end = longest_edge(shape["verts"])
    img.draw_arrow(int(start[0]), int(start[1]), int(end[0]), int(end[1]), color)

    cx, cy = int(shape["center"][0]), int(shape["center"][1])
    img.draw_cross(cx, cy, color, 6)
    img.draw_string(cx + 8, cy + 8, str(index), color, scale=1)


def format_shapes(shapes):
    """把每片结果拼成一行日志：颜色、顶点数、质心、朝向角、面积。"""
    parts = ["pieces=%d" % len(shapes)]
    for index, shape in enumerate(shapes):
        _, name = color_of(index)
        parts.append("#%d(%s) v=%d c=(%d,%d) a=%.1f area=%d" % (
            index, name, len(shape["verts"]),
            int(shape["center"][0]), int(shape["center"][1]),
            shape["angle"], int(shape["area"])))
    return " | ".join(parts)


def main():
    cam = camera.Camera(CAM_W, CAM_H)
    disp = display.Display()
    setup_camera(cam)

    shapes = []
    analyze_ms = 0.0
    last_analyze = 0.0
    frames = 0
    t_begin = time.time()
    while not app.need_exit():
        img = cam.read()
        now = time.time()

        if now - last_analyze >= ANALYZE_INTERVAL_SEC:
            t0 = time.time()
            shapes = find_piece_shapes(img)
            analyze_ms = (time.time() - t0) * 1000.0
            last_analyze = now

        for index, shape in enumerate(shapes):
            try:
                draw_shape(img, shape, index)
            except Exception as e:
                # 单片绘制失败不能让整个程序退出，现场不能因为显示异常停摆
                print("draw shape failed:", e)
        disp.show(img)

        frames += 1
        if now - t_begin >= PRINT_INTERVAL_SEC:
            print("fps %.1f | analyze %.0fms | %s" % (
                frames / (now - t_begin), analyze_ms, format_shapes(shapes)))
            frames = 0
            t_begin = now


main()
