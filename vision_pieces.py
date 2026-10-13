# vision_pieces.py - 碎片识别（分割 + 轮廓 + 凸包 + 顶点/质心/朝向）
#
# 依赖 maix 的 Image API 与 vision_geo 的几何工具。返回"缩小图分辨率"的像素坐标，
# 调用方用单应矩阵换算成毫米（见 main.py 的 pieces 模式）。
#
# 两段式处理（兼顾速度与精度）：
#   1. 全幅操作（二值化、找块、连通区隔离）在缩小的 work 图上做 -> 快；
#   2. 单片轮廓改到原图 crop 出来的小块上做 -> 准（像素密度翻倍，顶点才稳）。
#   早期只用缩小图，1 像素≈1.2mm，10~15mm 的短边只有十几个像素，顶点数会跳。
#
# 三条硬约束（改动前先看）：
#   * find_blobs 的 merge 必须为 False：它合并"包围盒相交"的色块，碎片斜放时会并肩；
#   * 不做逐像素生长的连通区隔离（那是掉到 1 帧/秒的原因）；
#   * 碎片一定是凸多边形，用凸包抹掉牌面图案啃出的缺口。
#
# 参数由调用方通过 configure() 覆盖，默认值是一套能跑的起点。

import math

from maix import image

import vision_geo

# ---- 可调参数（除特别说明外，单位都是"缩小图"像素）----
THRESHOLDS = [[20, 100, -30, 30, -30, 30]]              # 碎片颜色（LAB）
INVERT = False                                           # True：碎片比背景暗
PATCH_THRESHOLDS = [[50, 100, -128, 127, -128, 127]]     # 二值块上再取连通区的阈值
MIN_PIXELS = 200                                         # 小于此像素数不算碎片
MAX_PIXELS = 20000                                       # 大于此像素数多半是底板或背景
BBOX_PAD = 3                                             # 裁剪时包围盒外扩
CLOSE_SIZE = 0                                           # >0 闭运算补内部空洞（牌面图案用）
OPEN_SIZE = 0                                            # 开运算会削掉尖角，白片场景先关闭
EPSILON_RATIO = 0.01                                     # 简化阈值 = 周长 × 该比例
MIN_EPSILON = 2.0
MAX_EPSILON = 4.0
MERGE_DISTANCE = 8.0                                     # 相邻顶点近于此距离就合并
COMPENSATE_PX = 2.0                                      # 轮廓外扩补偿（阈值造成的内缩）


def configure(**kwargs):
    """覆盖模块级参数；只接受已存在的大写字段，参数名拼错会被明确报出来。"""
    for key, value in kwargs.items():
        if key in globals() and key.isupper():
            globals()[key] = value
        else:
            print("vision_pieces: 未知参数", key)


def build_mask(img):
    """二值化并按需做闭/开运算（在传入的图上做，可用于缩小图或局部小块）。"""
    mask_img = img.binary(THRESHOLDS, invert=INVERT, copy=True)
    if CLOSE_SIZE > 0:
        mask_img = mask_img.dilate(CLOSE_SIZE).erode(CLOSE_SIZE)
    if OPEN_SIZE > 0:
        mask_img = mask_img.erode(OPEN_SIZE).dilate(OPEN_SIZE)
    return mask_img


def pick_component(patch, seed_x, seed_y):
    """在局部二值块上再取一次连通区，剔掉框进来的邻片，返回离种子最近的 Blob。"""
    blobs = patch.find_blobs(PATCH_THRESHOLDS, pixels_threshold=MIN_PIXELS, merge=False)
    best = None
    best_distance = 0.0
    for blob in blobs:
        dx = blob[0] + blob[2] / 2.0 - seed_x
        dy = blob[1] + blob[3] / 2.0 - seed_y
        distance = dx * dx + dy * dy
        if best is None or distance < best_distance:
            best, best_distance = blob, distance
    return best


def find_piece_shapes(work, full, scale, width, height, roi=None, min_edge=0.0):
    """找出碎片，返回 [{"verts", "center", "angle", "area"}, ...]（缩小图分辨率坐标）。

    work: 缩小图（找块用，快）；full: 原图（裁轮廓用，准）；scale: 原图/缩小图 比例；
    width/height: 缩小图尺寸；roi 传 A4 包围盒可明显缩小搜索范围；
    min_edge: 最短边长度（缩小图像素），短于此的边会被合并掉，用来收拾假顶点。
    注意：只对每片的小块做二值化，避免整幅图多做一次 LAB 转换（那是帧率大头）。
    """
    blobs = work.find_blobs(THRESHOLDS, invert=INVERT, roi=roi if roi else [],
                            pixels_threshold=MIN_PIXELS, merge=False)
    shapes = []
    for blob in blobs:
        if blob.pixels() > MAX_PIXELS:
            continue        # 太大，多半是把底板或背景当成了一片

        x, y, w, h = blob[0], blob[1], blob[2], blob[3]
        x0 = max(0, x - BBOX_PAD)
        y0 = max(0, y - BBOX_PAD)
        x1 = min(width, x + w + BBOX_PAD)
        y1 = min(height, y + h + BBOX_PAD)
        if x1 - x0 < 5 or y1 - y0 < 5:
            continue

        coarse_mask = build_mask(work.crop(x0, y0, x1 - x0, y1 - y0))
        sub = pick_component(coarse_mask, blob.cx() - x0, blob.cy() - y0)
        if sub is None:
            continue
        sub_x, sub_y, sub_w, sub_h = sub[0], sub[1], sub[2], sub[3]
        if sub_w < 5 or sub_h < 5:
            continue

        # 按隔离结果去原图上裁同一块，得到高分辨率的轮廓
        fx0 = max(0, (x0 + sub_x) * scale)
        fy0 = max(0, (y0 + sub_y) * scale)
        fx1 = min(width * scale, (x0 + sub_x + sub_w) * scale)
        fy1 = min(height * scale, (y0 + sub_y + sub_h) * scale)
        if fx1 - fx0 < 5 or fy1 - fy0 < 5:
            continue
        fine_img = full.crop(fx0, fy0, fx1 - fx0, fy1 - fy0)
        fine_mask = fine_img.binary(THRESHOLDS, invert=INVERT, copy=True)
        if CLOSE_SIZE > 0:
            fine_mask = fine_mask.dilate(CLOSE_SIZE * scale).erode(CLOSE_SIZE * scale)
        if OPEN_SIZE > 0:
            fine_mask = fine_mask.erode(OPEN_SIZE * scale).dilate(OPEN_SIZE * scale)

        # 裁切框可能同时包含邻片；只选离种子最近的连通块，不改变掩膜形态。
        fine_seed_x = int(blob.cx() * scale - fx0)
        fine_seed_y = int(blob.cy() * scale - fy0)
        fine_sub = pick_component(fine_mask, fine_seed_x, fine_seed_y)
        if fine_sub is None or fine_sub[2] < 5 or fine_sub[3] < 5:
            continue
        fine_mask = fine_mask.crop(fine_sub[0], fine_sub[1], fine_sub[2], fine_sub[3])
        fx0 += fine_sub[0]
        fy0 += fine_sub[1]

        local = image.image2cv(fine_mask)[:, :, 0] > 128
        contour = vision_geo.scan_contour(local)
        if len(contour) < 8:
            continue

        absolute = [(px + fx0, py + fy0) for px, py in contour]
        hull = vision_geo.convex_hull(absolute)
        if len(hull) < 3:
            continue

        epsilon = min(max(vision_geo.contour_perimeter(hull) * EPSILON_RATIO,
                          MIN_EPSILON * scale), MAX_EPSILON * scale)
        verts = vision_geo.merge_close_vertices(
            vision_geo.simplify_closed(hull, epsilon), MERGE_DISTANCE * scale)
        if len(verts) < 3:
            continue
        if min_edge > 0:      # 按题目约束收拾假顶点：明显短于 2cm 的边合并掉
            verts = vision_geo.merge_short_edges(verts, min_edge * scale)
        verts = vision_geo.expand_polygon(verts, COMPENSATE_PX * scale)
        verts = [(px / scale, py / scale) for px, py in verts]   # 回到缩小图分辨率

        center, area = vision_geo.polygon_centroid(verts)
        angle, _, _, _ = vision_geo.longest_edge(verts)
        shapes.append({"verts": verts, "center": center, "angle": angle, "area": area})
    return shapes


def probe(work, roi=None):
    """诊断用：返回阈值内的候选块数与最大块像素数。

    用来区分"分割根本没分出东西"（候选=0）还是"分出来了但被尺寸过滤掉"（候选有、最大块超限）。
    """
    blobs = work.find_blobs(THRESHOLDS, invert=INVERT, roi=roi if roi else [],
                            pixels_threshold=max(1, MIN_PIXELS // 4), merge=False)
    biggest = 0
    for blob in blobs:
        biggest = max(biggest, blob.pixels())
    return len(blobs), biggest


def format_piece_mm(index, verts_mm):
    """把一片的毫米位姿整理成一行：顶点数、各边长、质心、朝向角、面积。"""
    center, area = vision_geo.polygon_centroid(verts_mm)
    angle, _, _, _ = vision_geo.longest_edge(verts_mm)
    count = len(verts_mm)
    edges = []
    for i in range(count):
        x1, y1 = verts_mm[i]
        x2, y2 = verts_mm[(i + 1) % count]
        edges.append(math.hypot(x2 - x1, y2 - y1))
    return "#%d v=%d 边长[%s] 质心=(%.1f,%.1f) 角度=%.1f 面积=%.0fmm2" % (
        index, count, ",".join("%.1f" % e for e in edges),
        center[0], center[1], angle, area)


def pieces_json(a4_mm, px_to_mm, shapes, targets=None, assembly="none"):
    """把识别结果打包成可视化页面（viz_assembly.html）能直接读的一行 JSON。

    a4_mm 是 A4 的毫米尺寸 (宽, 高)，px_to_mm 是标定得到的单应矩阵。
    targets 是 {id: {target, rotate}}；没有求解结果时页面只画初始位置。
    输出的 verts 已经是毫米坐标、A4 左上角为原点，页面可以直接画。
    """
    targets = targets or {}
    items = []
    for index, shape in enumerate(shapes):
        verts = [vision_geo.apply_homography(px_to_mm, x, y) for x, y in shape["verts"]]
        text = ",".join("[%.1f,%.1f]" % (px, py) for px, py in verts)
        item = '{"id":%d,"verts":[%s]' % (index, text)
        target = targets.get(index)
        if target:
            target_text = ",".join("[%.1f,%.1f]" % (px, py)
                                   for px, py in target["target"])
            item += ',"target":[%s],"rotate":%.1f' % (target_text, target["rotate"])
        item += "}"
        items.append(item)
    return '{"a4":[%g,%g],"assembly":"%s","pieces":[%s]}' % (
        a4_mm[0], a4_mm[1], assembly, ",".join(items))
