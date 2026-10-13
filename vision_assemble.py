# vision_assemble.py - 拼图求解（纯几何，不依赖 maix，可离线验证）
#
# 输入：每片的毫米多边形（顶点序列，任意摆放姿态）
# 输出：每片的目标多边形与需要旋转的角度；解不出来返回 None
#
# 思路（针对题目约束：片数≤4、每片边数≤5、每条边≥2cm）：
#   1. 相邻碎片沿同一条切割线分开，所以两片的边共线且端点对齐——注意**不要求等长**，
#      一条长边可能同时贴着另外两片的短边，所以用"顶点对齐 + 边共线 + 有实际接触长度"当候选；
#   2. 从面积最大的一片开始，把后续片贴到已放好的边上，递归放完（重叠用分离轴定理判）；
#   3. 剪枝：部分拼装的凸包面积不能超过所有碎片的总面积（最终矩形就那么大）；
#   4. 拼完后用凸包判矩形：外轮廓是 4 个直角的四边形，且各片面积和等于凸包面积（正好铺满）；
#   5. 找到第一个满铺矩形就返回，并把它摆到 A4 下半区；不追多解，避免板端超时。

import math

import vision_geo

MAX_STEPS = 200000        # 兜底：板端不能因为搜索失控卡死
MAX_SOLUTIONS = 1         # 找到一个满铺矩形就够搬运；继续找多解会明显拖慢板端
CONTACT_RATIO = 0.5       # 真实相邻边至少覆盖较短边的一半；用于砍掉“只碰角”的伪候选
MIN_CONTACT_MM = 1.0


def _edges(verts):
    """返回 [(长度, 起点, 终点), ...]。"""
    result = []
    count = len(verts)
    for i in range(count):
        x1, y1 = verts[i]
        x2, y2 = verts[(i + 1) % count]
        result.append((math.hypot(x2 - x1, y2 - y1), (x1, y1), (x2, y2)))
    return result


def _rotate(points, deg):
    rad = math.radians(deg)
    cos_a, sin_a = math.cos(rad), math.sin(rad)
    return [(x * cos_a - y * sin_a, x * sin_a + y * cos_a) for x, y in points]


def _translate(points, dx, dy):
    return [(x + dx, y + dy) for x, y in points]


def _hull(polys):
    return vision_geo.convex_hull([point for poly in polys for point in poly])


def _hull_area(polys):
    hull = _hull(polys)
    return vision_geo.polygon_centroid(hull)[1] if len(hull) >= 3 else 0.0


def polygons_overlap(a, b, margin=0.8):
    """两个凸多边形是否真的重叠（共边接触不算）。分离轴定理。"""
    for poly in (a, b):
        count = len(poly)
        for i in range(count):
            x1, y1 = poly[i]
            x2, y2 = poly[(i + 1) % count]
            nx, ny = -(y2 - y1), (x2 - x1)
            length = math.hypot(nx, ny)
            if length < 1e-9:
                continue
            nx, ny = nx / length, ny / length
            amin = min(nx * px + ny * py for px, py in a)
            amax = max(nx * px + ny * py for px, py in a)
            bmin = min(nx * px + ny * py for px, py in b)
            bmax = max(nx * px + ny * py for px, py in b)
            if amax - margin <= bmin or bmax - margin <= amin:
                return False
    return True


def _candidate_placements(placed_polys, local):
    """所有"顶点对齐 + 边共线 + 接触长度>1mm"的摆放候选，去重后返回 [(顶点, 旋转角)]。"""
    local_edges = _edges(local)
    seen = set()
    out = []
    for placed in placed_polys:
        for edge in _edges(placed):
            (px1, py1), (px2, py2) = edge[1], edge[2]
            base = math.degrees(math.atan2(py2 - py1, px2 - px1))
            ux, uy = px2 - px1, py2 - py1
            ulen = math.hypot(ux, uy)
            if ulen < 1e-9:
                continue
            ux, uy = ux / ulen, uy / ulen
            span_a = sorted((px1 * ux + py1 * uy, px2 * ux + py2 * uy))
            for local_edge in local_edges:
                (qx1, qy1), (qx2, qy2) = local_edge[1], local_edge[2]
                here = math.degrees(math.atan2(qy2 - qy1, qx2 - qx1))
                for flip in (0.0, 180.0):
                    theta = base - here + flip
                    rotated = _rotate(local, theta)
                    rq1 = _rotate([(qx1, qy1)], theta)[0]
                    rq2 = _rotate([(qx2, qy2)], theta)[0]
                    span_b = sorted((rq1[0] * ux + rq1[1] * uy, rq2[0] * ux + rq2[1] * uy))
                    contact = min(span_a[1], span_b[1]) - max(span_a[0], span_b[0])
                    min_edge = min(ulen, local_edge[0])
                    if contact < max(MIN_CONTACT_MM, min_edge * CONTACT_RATIO):
                        continue        # 只碰角或接触太短，不算真实相邻边
                    for anchor, target in ((rq1, (px1, py1)), (rq1, (px2, py2)),
                                           (rq2, (px1, py1)), (rq2, (px2, py2))):
                        verts = _translate(rotated, target[0] - anchor[0], target[1] - anchor[1])
                        key = tuple((round(x, 1), round(y, 1)) for x, y in verts)
                        if key in seen:
                            continue
                        seen.add(key)
                        out.append((verts, theta))
    return out


def _rectangle_score(polys, angle_tol=0.08, area_tol=0.03):
    """用凸包判矩形（与拼装朝向无关）：4 个直角 + 各片正好铺满。返回 (是否, 长宽比)。"""
    hull = _hull(polys)
    if len(hull) != 4:
        return False, 0.0
    for i in range(4):
        x0, y0 = hull[i - 1]
        x1, y1 = hull[i]
        x2, y2 = hull[(i + 1) % 4]
        v1 = (x0 - x1, y0 - y1)
        v2 = (x2 - x1, y2 - y1)
        denom = math.hypot(v1[0], v1[1]) * math.hypot(v2[0], v2[1])
        if denom < 1e-9 or abs((v1[0] * v2[0] + v1[1] * v2[1]) / denom) > angle_tol:
            return False, 0.0
    hull_area = vision_geo.polygon_centroid(hull)[1]
    total = sum(vision_geo.polygon_centroid(poly)[1] for poly in polys)
    if hull_area < 1e-6 or abs(total - hull_area) / hull_area > area_tol:
        return False, 0.0
    side1 = math.hypot(hull[1][0] - hull[0][0], hull[1][1] - hull[0][1])
    side2 = math.hypot(hull[2][0] - hull[1][0], hull[2][1] - hull[1][1])
    if min(side1, side2) < 1e-6:
        return False, 0.0
    return True, max(side1, side2) / min(side1, side2)


def _normalize_angle(deg):
    """角度归一化到 (-180, 180]，便于显示与下位机执行。"""
    angle = math.fmod(deg + 180.0, 360.0)
    if angle < 0:
        angle += 360.0
    return angle - 180.0


def _has_axis_edge(verts, axis_deg, tol_deg=6.0):
    """该片是否至少有 1 条边平行或垂直于给定轴向。

    题目保证每片至少有 1 条边落在目标矩形外边界上，所以这条边必然平行于矩形的某一轴；
    用它剪枝能砍掉大量无效摆法。
    """
    for edge in _edges(verts):
        (x1, y1), (x2, y2) = edge[1], edge[2]
        ang = math.degrees(math.atan2(y2 - y1, x2 - x1)) - axis_deg
        if abs(((ang + 45.0) % 90.0) - 45.0) <= tol_deg:
            return True
    return False


def solve(pieces, a4_mm=(297.0, 210.0)):
    """求解拼图：pieces 为 [{"id", "verts"}]（毫米），返回 [{id, target, rotate}] 或 None。"""
    if not pieces:
        return None

    local_shapes = []
    for piece in pieces:
        center = vision_geo.polygon_centroid(piece["verts"])[0]
        local_shapes.append({"id": piece["id"],
                             "local": [(x - center[0], y - center[1]) for x, y in piece["verts"]]})
    total_area = sum(vision_geo.polygon_centroid(item["local"])[1] for item in local_shapes)

    # 先放面积最大的那片当基准，能显著减少搜索分支
    order = sorted(range(len(local_shapes)),
                   key=lambda i: -vision_geo.polygon_centroid(local_shapes[i]["local"])[1])
    solutions = []
    steps = [0]

    def search(placed, remaining, axis):
        steps[0] += 1
        if steps[0] > MAX_STEPS or len(solutions) >= MAX_SOLUTIONS:
            return
        placed_polys = [item["verts"] for item in placed]
        if not remaining:
            ok, ratio = _rectangle_score(placed_polys)
            if ok and 0.85 <= ratio <= 2.6:
                solutions.append(list(placed))
            return
        for index in [i for i in order if i in remaining]:
            local = local_shapes[index]["local"]
            for verts, theta in _candidate_placements(placed_polys, local):
                if not _has_axis_edge(verts, axis):
                    continue
                if any(polygons_overlap(verts, other) for other in placed_polys):
                    continue
                # 部分拼装的凸包不可能比最终矩形还大，超过总面积就说明走偏了
                if _hull_area(placed_polys + [verts]) > total_area * 1.02:
                    continue
                placed.append({"id": index, "verts": verts, "rotate": theta})
                remaining.discard(index)
                search(placed, remaining, axis)
                remaining.add(index)
                placed.pop()
                if steps[0] > MAX_STEPS or len(solutions) >= MAX_SOLUTIONS:
                    return

    seed = order[0]
    seed_local = local_shapes[seed]["local"]
    # 基准片的每条边都可能是矩形的一个轴向，逐个假设再搜，能大幅剪枝
    for seed_edge in _edges(seed_local):
        axis = math.degrees(math.atan2(seed_edge[2][1] - seed_edge[1][1],
                                       seed_edge[2][0] - seed_edge[1][0]))
        search([{"id": seed, "verts": seed_local, "rotate": 0.0}],
               set(range(len(local_shapes))) - {seed}, axis)
        if len(solutions) >= MAX_SOLUTIONS or steps[0] > MAX_STEPS:
            break
    if not solutions:
        return None

    best = None
    for solution in solutions:
        ratio = _rectangle_score([item["verts"] for item in solution])[1]
        score = abs(math.log(ratio / 1.5))
        if best is None or score < best[0]:
            best = (score, solution)

    chosen = best[1]
    return _place_on_a4(chosen, a4_mm)


def _place_on_a4(chosen, a4_mm):
    """把拼好的矩形转正（轴对齐）并摆到 A4 下半区居中，返回 [{id, target, rotate}]。"""
    polys = [item["verts"] for item in chosen]
    hull = _hull(polys)
    center = vision_geo.polygon_centroid(hull)[0]
    angle = math.degrees(math.atan2(hull[1][1] - hull[0][1], hull[1][0] - hull[0][0]))
    if angle > 45.0:
        angle -= 90.0
    if angle < -45.0:
        angle += 90.0

    aligned = []
    for item in chosen:
        moved = _translate(_rotate(item["verts"], -angle), -center[0], -center[1])
        aligned.append({"id": item["id"], "verts": moved,
                        "rotate": _normalize_angle(item["rotate"] - angle)})

    xs = [point[0] for item in aligned for point in item["verts"]]
    ys = [point[1] for item in aligned for point in item["verts"]]
    dx = (a4_mm[0] - (max(xs) - min(xs))) / 2.0 - min(xs)
    margin = 10.0
    dy = a4_mm[1] * 0.68 - (min(ys) + max(ys)) / 2.0
    dy = max(margin - min(ys), min(dy, a4_mm[1] - margin - max(ys)))
    return [{"id": item["id"], "target": _translate(item["verts"], dx, dy),
             "rotate": item["rotate"]} for item in aligned]


def solve_for_shapes(shapes, px_to_mm, a4_mm):
    """把识别结果（缩小图像素坐标）换算成毫米后求解，返回 {id: {target, rotate}} 或 None。"""
    pieces = []
    for index, shape in enumerate(shapes):
        verts = [vision_geo.apply_homography(px_to_mm, x, y) for x, y in shape["verts"]]
        pieces.append({"id": index, "verts": verts})
    result = solve(pieces, a4_mm)
    if not result:
        return None
    return {item["id"]: item for item in result}


def preview_for_shapes(shapes, px_to_mm, a4_mm, gap_mm=8.0):
    """求解失败时的可视化降级：把各片平移到 A4 下半区排成一行，仅用于画搬运轨迹。

    该结果不是题目要求的真实拼图解，调用方必须在 JSON 中把 assembly 标为 "preview"。
    """
    pieces = []
    for index, shape in enumerate(shapes):
        verts = [vision_geo.apply_homography(px_to_mm, x, y) for x, y in shape["verts"]]
        center, area = vision_geo.polygon_centroid(verts)
        xs = [point[0] for point in verts]
        ys = [point[1] for point in verts]
        pieces.append({"id": index, "verts": verts, "center": center, "area": area,
                       "width": max(xs) - min(xs), "height": max(ys) - min(ys)})
    if not pieces:
        return {}

    total_width = sum(item["width"] for item in pieces) + gap_mm * (len(pieces) - 1)
    scale = min(1.0, (a4_mm[0] - 24.0) / max(total_width, 1e-6))
    gap = gap_mm * scale
    x = (a4_mm[0] - (total_width * scale)) / 2.0
    target_y = a4_mm[1] * 0.62
    result = {}
    for item in pieces:
        target_center = (x + item["width"] * scale / 2.0, target_y)
        dx = target_center[0] - item["center"][0]
        dy = target_center[1] - item["center"][1]
        target = [(px + dx, py + dy) for px, py in item["verts"]]
        result[item["id"]] = {"id": item["id"], "target": target, "rotate": 0.0}
        x += item["width"] * scale + gap
    return result


def layout_for_shapes(shapes, px_to_mm, a4_mm):
    """返回 (目标字典, 状态)。状态为 "solved" 或 "preview"。"""
    try:
        solved = solve_for_shapes(shapes, px_to_mm, a4_mm)
    except Exception as e:
        print("assemble solve failed:", e)
        solved = None
    if solved:
        return solved, "solved"
    return preview_for_shapes(shapes, px_to_mm, a4_mm), "preview"
