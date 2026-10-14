# vision_geo.py - 视觉几何工具（纯计算，不依赖 maix，便于离线测试与多文件复用）
#
# 提供：轮廓扫描、多边形简化与合并、凸包、质心、最长边角度、外扩补偿、单应矩阵。
# 约定：所有函数里的"像素"都指调用方给定的坐标系（工作分辨率或原分辨率，由调用方保证一致）。

import math

import numpy as np


def scan_contour(mask):
    """逐行扫描取轮廓：碎片是凸的，每行前景连续，所以左右极值即完整轮廓，
    一次 numpy argmax 算完整块（非凸图形需换成沿边界追踪）。"""
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


def merge_short_edges(points, min_length):
    """反复合并最短边，直到每条边都不短于 min_length（用题目约束收拾假顶点）。

    题目规定现场碎片每条边不小于 2cm；明显短于它的边基本都是顶点提歪产生的假边，
    按物理尺寸合并比调几何简化阈值稳得多。
    """
    merged = list(points)
    while len(merged) > 3:
        count = len(merged)
        shortest, index = None, 0
        for i in range(count):
            x1, y1 = merged[i]
            x2, y2 = merged[(i + 1) % count]
            length = math.hypot(x2 - x1, y2 - y1)
            if shortest is None or length < shortest:
                shortest, index = length, i
        if shortest >= min_length:
            break
        first = merged[index]
        second = merged[(index + 1) % count]
        middle = ((first[0] + second[0]) / 2.0, (first[1] + second[1]) / 2.0)
        if index + 1 < count:
            merged = merged[:index] + [middle] + merged[index + 2:]
        else:
            merged = [middle] + merged[1:count - 1]
    return merged


def convex_hull(points):
    """凸包（Andrew 单调链）：碎片是矩形直线切割得到的，一定是凸多边形，
    凸包能一次抹掉图案缺口与边界锯齿。"""
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
    """按顶点序列计算多边形面积质心，返回 ((cx, cy), 面积)。"""
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


def expand_polygon(points, distance):
    """把顶点沿质心方向外扩 distance，补偿阈值只保留亮边造成的内缩。
    属近似均匀外扩（对角点略多、对长边中点略少），抵消 1~3 像素的系统偏差够用。"""
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


def order_quad(points):
    """把四个点排成 左上 / 右上 / 右下 / 左下。

    判定用 x+y 与 x-y：左上 x+y 最小，右下 x+y 最大，右上 x-y 最大，左下最小。
    要求目标在画面里的旋转不超过 45 度（正常摆放都满足）。
    """
    if len(points) != 4:
        return list(points)
    ordered = sorted(points, key=lambda p: p[0] + p[1])
    top_left, bottom_right = ordered[0], ordered[-1]
    rest = [p for p in points if p != top_left and p != bottom_right]
    if len(rest) != 2:
        return list(points)
    if (rest[0][0] - rest[0][1]) > (rest[1][0] - rest[1][1]):
        return [top_left, rest[0], bottom_right, rest[1]]
    return [top_left, rest[1], bottom_right, rest[0]]


def compute_homography(src, dst):
    """由 4 组对应点解出单应矩阵（src -> dst），返回 3x3 numpy 数组；失败返回 None。

    单应矩阵 8 个自由度，4 组点正好给 8 个方程，直接解线性方程组。
    传入 (像素四角, 毫米四角) 得到"像素->毫米"，反过来则得到"毫米->像素"。
    """
    if len(src) != 4 or len(dst) != 4:
        return None
    rows = []
    values = []
    for (x, y), (u, v) in zip(src, dst):
        rows.append([x, y, 1.0, 0.0, 0.0, 0.0, -u * x, -u * y])
        values.append(float(u))
        rows.append([0.0, 0.0, 0.0, x, y, 1.0, -v * x, -v * y])
        values.append(float(v))
    try:
        solved = np.linalg.solve(np.array(rows, dtype=float), np.array(values, dtype=float))
    except Exception as e:
        print("homography solve failed:", e)
        return None
    return np.array([[solved[0], solved[1], solved[2]],
                     [solved[3], solved[4], solved[5]],
                     [solved[6], solved[7], 1.0]])


def apply_homography(matrix, x, y):
    """用单应矩阵把点 (x, y) 映射到目标坐标系，返回 (u, v)。"""
    if matrix is None:
        return (x, y)
    denominator = matrix[2][0] * x + matrix[2][1] * y + matrix[2][2]
    if abs(denominator) < 1e-9:
        return (x, y)
    return ((matrix[0][0] * x + matrix[0][1] * y + matrix[0][2]) / denominator,
            (matrix[1][0] * x + matrix[1][1] * y + matrix[1][2]) / denominator)


def polygon_inside_rect(px_to_mm, points, mm_corners, margin=3.0):
    """判断像素多边形是否完整落在 A4 毫米矩形内（margin 允许少量边界误差）。"""
    xs = [point[0] for point in mm_corners]
    ys = [point[1] for point in mm_corners]
    min_x, max_x = min(xs) - margin, max(xs) + margin
    min_y, max_y = min(ys) - margin, max(ys) + margin
    for x, y in points:
        u, v = apply_homography(px_to_mm, x, y)
        if not (min_x <= u <= max_x and min_y <= v <= max_y):
            return False
    return True
