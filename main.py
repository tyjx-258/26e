# main.py - MaixCAM Pro 视觉端主程序（唯一入口）
#
# MaixVision 只能"运行完整项目"，所以各流程都收在这里，用 RUN_MODE 切换：
#   "a4_calib" - A4 四角标定：检出四角算 像素<->毫米 单应矩阵，并画 100mm 参考线核对
#   "pieces"   - 碎片识别：用四角标定把碎片顶点换算成毫米，打印每片位姿
# 运行：MaixVision 打开本工程文件夹 -> 运行完整项目。
# 依赖：vision_geo.py（纯几何）、vision_pieces.py（碎片识别）；曝光标定见 calib_exposure.py。
# 硬件假设：MaixCAM / MaixCAM Pro 本体；未接下位机时保持 USE_UART = False。

import math
import time

from maix import app, camera, display, image, pinmap, uart, err

import vision_geo
import vision_pieces

# ---- 运行模式 ----
RUN_MODE = "pieces"              # "a4_calib" | "pieces"

# ---- 串口（对接下位机）----
USE_UART = False                 # 未接 STM32 时保持 False；接上后改 True
UART_DEV = "/dev/ttyS1"          # 不要用 /dev/ttyS0：它会输出启动日志，A16 被拉低还可能起不来
UART_TX_PIN = "A19"
UART_RX_PIN = "A18"
UART_BAUD = 115200

# ---- 相机 ----
CAM_W = 640
CAM_H = 480
EXPOSURE_US = 26000              # 现场实测值；换环境用 calib_exposure.py 重新定档
GAIN_VALUE = 500
WARMUP_SEC = 0.5                 # 空跑少量帧让画面稳定，再锁白平衡
LOCK_WHITE_BALANCE = True
WB_GAIN_FALLBACK = [0.134, 0.0625, 0.0625, 0.1239]   # 官方给出的 MaixCAM 初始白平衡增益

# ---- 工作分辨率 ----
WORK_SCALE = 2                   # 标定与识别都在 1/2 分辨率上做，坐标最后乘回来
WORK_W = CAM_W // WORK_SCALE
WORK_H = CAM_H // WORK_SCALE

# ---- A4 四角标定 ----
A4_MM_SHORT = 210.0              # A4 短边（毫米）
A4_MM_LONG = 297.0               # A4 长边（毫米）
A4_INVERT = False                # True：A4 比背景暗（深色 A4 + 浅色垫子）
A4_CALIB_THRESHOLDS = [[55, 100, -128, 127, -128, 127]]
A4_MIN_AREA_RATIO = 0.15         # A4 至少要占画面这么大比例，用来与碎片区分
A4_CALIB_MIN_PIXELS = 2000       # 工作分辨率下的最小像素数（兜底）
A4_BBOX_PAD = 3
A4_CLOSE_SIZE = 3                # 对 A4 掩膜做闭运算，补上被阴影吃掉的边缘断口
EPSILON_RATIO = 0.01             # 简化阈值 = 轮廓周长 × 该比例
MIN_EPSILON = 2.0
MAX_EPSILON = 4.0
MERGE_DISTANCE = 8.0             # 相邻顶点近于此距离就合并成一个
REFERENCE_MM = 100.0             # 参考线长度（毫米），用尺子核对标定
QUAD_SMOOTH_ALPHA = 0.25         # 四角时序平滑系数：越小越稳、越大跟得越快
QUAD_TOL_PX = 12                 # 与锁定四角的平均偏差超过它就认为纸张在动
QUAD_SWITCH_FRAMES = 4           # 连续多少次重算都偏离才切换锁定四角（约 2 秒）
# 固定标定四角（工作分辨率像素，顺序 左上/右上/右下/左下）：
# 相机不动时用 a4_calib 量一次，把日志那行粘到这里；之后 pieces 直接用它，不再检测 A4。
A4_QUAD_OVERRIDE = [(31, 8), (277, 30), (270, 190), (31, 198)]

# ---- 碎片识别参数（工作分辨率像素）----
PIECE_THRESHOLDS = [[30, 100, -128, 127, -128, 127]]   # 碎片颜色（LAB），下限跟现场实测一致
PIECE_INVERT = False             # True：碎片比背景暗
PIECE_MIN_PIXELS = 200           # 小于此像素数不算碎片
PIECE_MAX_PIXELS = 20000         # 大于此像素数多半是底板或背景
MIN_EDGE_MM = 10.0               # 短于此长度的边判为假边并合并（题目要求现场碎片边 ≥20mm）
PIECE_COLORS = [image.COLOR_RED, image.COLOR_GREEN, image.COLOR_BLUE,
                image.COLOR_YELLOW, image.COLOR_PURPLE, image.COLOR_ORANGE]

# ---- 运行开关 ----
VERBOSE = True                   # True：每秒打印运行状态，便于现场调试
STATUS_REPORT_SEC = 1.0
ANALYZE_INTERVAL_SEC = 0.8       # 重算间隔：中间帧复用上次结果，重活不必每帧做


def open_uart():
    """按配置打开与下位机的串口；失败返回 None，让视觉功能继续可用。"""
    err.check_raise(pinmap.set_pin_function(UART_TX_PIN, "UART1_TX"), "set UART1 TX failed")
    err.check_raise(pinmap.set_pin_function(UART_RX_PIN, "UART1_RX"), "set UART1 RX failed")
    try:
        return uart.UART(UART_DEV, UART_BAUD)
    except Exception as e:
        print("uart open failed:", e)
        return None


def read_camera_params(cam):
    """读取当前曝光与增益，仅用于记录；取不到时返回 (None, None)。"""
    try:
        return cam.exposure(), cam.gain()
    except Exception as e:
        print("read camera params failed:", e)
        return None, None


def lock_white_balance(cam):
    """锁定白平衡；接口不可用时只提示并保持自动白平衡，不影响主流程。"""
    try:
        gains = cam.set_wb_gain()
        gains = list(gains) if gains else list(WB_GAIN_FALLBACK)
        cam.set_wb_gain(gains)
        print("white balance locked:", [round(float(g), 4) for g in gains])
    except Exception as e:
        print("white balance lock skipped:", e)


def setup_camera(cam):
    """固定曝光/增益/白平衡，避免自动调节随时间漂移；任一步失败只提示不中断。"""
    t0 = time.time()
    while not app.need_exit() and time.time() - t0 < WARMUP_SEC:
        cam.read()

    try:
        if EXPOSURE_US > 0:
            cam.exposure(EXPOSURE_US)
        if GAIN_VALUE > 0:
            cam.gain(GAIN_VALUE)
    except Exception as e:
        print("exposure/gain lock skipped:", e)

    if LOCK_WHITE_BALANCE:
        lock_white_balance(cam)
    return read_camera_params(cam)


def build_work_image(img):
    """按 WORK_SCALE 缩小图像；标定与识别都在小图上做，省算力。"""
    if WORK_SCALE <= 1:
        return img
    return img.resize(WORK_W, WORK_H)


def find_a4_quad(work):
    """在缩小图上找 A4 纸张，返回四个角（工作分辨率坐标，顺序 左上/右上/右下/左下）。

    A4 是画面里最大的亮块（白纸放深色垫上）；深色 A4 放浅色垫上时把 A4_INVERT 置 True。
    四角用 轮廓 -> 凸包 -> 简化 -> 顶点合并 得到，恰好 4 个才认，否则返回 None。
    """
    mask_img = work.binary(A4_CALIB_THRESHOLDS, invert=A4_INVERT, copy=True)
    if A4_CLOSE_SIZE > 0:        # 闭运算：补上被阴影吃掉的边缘断口，四角更稳
        mask_img = mask_img.dilate(A4_CLOSE_SIZE).erode(A4_CLOSE_SIZE)
    min_pixels = max(A4_CALIB_MIN_PIXELS, int(WORK_W * WORK_H * A4_MIN_AREA_RATIO))
    blobs = work.find_blobs(A4_CALIB_THRESHOLDS, invert=A4_INVERT,
                            pixels_threshold=min_pixels, merge=False)
    if not blobs:
        return None

    blob = max(blobs, key=lambda item: item.pixels())
    x, y, w, h = blob[0], blob[1], blob[2], blob[3]
    x0 = max(0, x - A4_BBOX_PAD)
    y0 = max(0, y - A4_BBOX_PAD)
    x1 = min(WORK_W, x + w + A4_BBOX_PAD)
    y1 = min(WORK_H, y + h + A4_BBOX_PAD)
    if x1 - x0 < 5 or y1 - y0 < 5:
        return None

    patch = mask_img.crop(x0, y0, x1 - x0, y1 - y0)
    local = image.image2cv(patch)[:, :, 0] > 128
    contour = vision_geo.scan_contour(local)
    if len(contour) < 8:
        return None

    absolute = [(px + x0, py + y0) for px, py in contour]
    hull = vision_geo.convex_hull(absolute)
    epsilon = min(max(vision_geo.contour_perimeter(hull) * EPSILON_RATIO,
                      MIN_EPSILON), MAX_EPSILON)
    quad = vision_geo.merge_close_vertices(
        vision_geo.simplify_closed(hull, epsilon), MERGE_DISTANCE)
    if len(quad) != 4:
        return None
    return vision_geo.order_quad(quad)


def mm_corners_for(quad):
    """按四角在画面里的长宽比判断 A4 横放还是竖放，返回 (毫米四角, 方向名)。

    横竖弄反会让标定尺寸差 297/210 ≈ 1.41 倍，参考线就会一长一短。
    返回顺序与 quad 一致：左上 / 右上 / 右下 / 左下。
    """
    (x0, y0), (x1, y1), _, (x3, y3) = quad
    width = math.hypot(x1 - x0, y1 - y0)
    height = math.hypot(x3 - x0, y3 - y0)
    if width >= height:      # 长边水平：横放
        return ([(0.0, 0.0), (A4_MM_LONG, 0.0), (A4_MM_LONG, A4_MM_SHORT), (0.0, A4_MM_SHORT)],
                "landscape")
    return ([(0.0, 0.0), (A4_MM_SHORT, 0.0), (A4_MM_SHORT, A4_MM_LONG), (0.0, A4_MM_LONG)],
            "portrait")


def draw_calibration(img, quad, mm_corners, mm_to_px):
    """画 A4 四边与角点序号，再在纸张中心画水平、垂直两条 REFERENCE_MM 毫米参考线。

    mm_to_px 由调用方预先算好：单应矩阵解算较重，不必每帧重算。
    """
    pixels = [(px * WORK_SCALE, py * WORK_SCALE) for px, py in quad]
    for i in range(4):
        x1, y1 = pixels[i]
        x2, y2 = pixels[(i + 1) % 4]
        img.draw_line(int(x1), int(y1), int(x2), int(y2), image.COLOR_GREEN)
        img.draw_string(int(x1) + 6, int(y1) + 6, str(i), image.COLOR_GREEN, scale=1)

    half_w = mm_corners[1][0] / 2.0
    half_h = mm_corners[2][1] / 2.0
    half_ref = REFERENCE_MM / 2.0
    lines = (((half_w - half_ref, half_h), (half_w + half_ref, half_h)),
             ((half_w, half_h - half_ref), (half_w, half_h + half_ref)))
    for start_mm, end_mm in lines:
        ax, ay = vision_geo.apply_homography(mm_to_px, start_mm[0], start_mm[1])
        bx, by = vision_geo.apply_homography(mm_to_px, end_mm[0], end_mm[1])
        img.draw_line(int(ax * WORK_SCALE), int(ay * WORK_SCALE),
                      int(bx * WORK_SCALE), int(by * WORK_SCALE), image.COLOR_RED)


def quad_offset(a, b):
    """两组四角的平均点距（工作分辨率像素），用来判断纸张有没有真的移动。"""
    total = 0.0
    for (x1, y1), (x2, y2) in zip(a, b):
        total += math.hypot(x2 - x1, y2 - y1)
    return total / len(a)


def smooth_quad(old, new, alpha):
    """四角做指数平滑：压掉逐帧抖动，同时保留跟随纸张移动的能力。"""
    return [((ox * (1.0 - alpha) + nx * alpha), (oy * (1.0 - alpha) + ny * alpha))
            for (ox, oy), (nx, ny) in zip(old, new)]


def run_a4_calib(cam, disp):
    """A4 四角标定：画四角与参考线；红线用尺子量，应正好 REFERENCE_MM 毫米。
    重算每 ANALYZE_INTERVAL_SEC 一次，中间帧复用；四角时序平滑，连续多帧偏离才认移动。"""
    frames = 0
    t_begin = time.time()
    locked = None
    suspect = 0
    orientation = "-"
    mm_corners = None
    mm_to_px = None
    last_analyze = 0.0
    while not app.need_exit():
        img = cam.read()
        now = time.time()

        if now - last_analyze >= ANALYZE_INTERVAL_SEC:
            last_analyze = now
            live = find_a4_quad(build_work_image(img))
            if live is not None:
                if locked is None:
                    locked = list(live)
                elif quad_offset(locked, live) <= QUAD_TOL_PX:
                    locked = smooth_quad(locked, live, QUAD_SMOOTH_ALPHA)
                    suspect = 0
                else:
                    suspect += 1
                    if suspect >= QUAD_SWITCH_FRAMES:   # 连续偏离够多，判定纸张真的动了
                        locked = list(live)
                        suspect = 0
            if locked is not None:
                mm_corners, orientation = mm_corners_for(locked)
                mm_to_px = vision_geo.compute_homography(mm_corners, locked)

        if locked is not None:
            draw_calibration(img, locked, mm_corners, mm_to_px)
        disp.show(img)

        frames += 1
        if now - t_begin >= STATUS_REPORT_SEC:
            if locked is None:
                print("fps %.1f | A4 未检出：检查 A4 与背景对比、A4_INVERT、阈值" %
                      (frames / (now - t_begin)))
            else:
                coords = " ".join("(%d,%d)" % (int(px * WORK_SCALE), int(py * WORK_SCALE))
                                  for px, py in locked)
                print("fps %.1f | 纸张方向=%s | A4 四角(原图,左上起顺时针): %s | 红线应正好 %.0fmm" %
                      (frames / (now - t_begin), orientation, coords, REFERENCE_MM))
                print("   固定标定用（工作分辨率，可直接粘到 main.py）: A4_QUAD_OVERRIDE = [%s]" %
                      ", ".join("(%d, %d)" % (int(px), int(py)) for px, py in locked))
            frames = 0
            t_begin = now


def a4_roi(quad, margin=6):
    """由 A4 四角算出包围盒，作为碎片搜索的 ROI，能省不少算力。"""
    xs = [point[0] for point in quad]
    ys = [point[1] for point in quad]
    x0 = max(0, int(min(xs)) - margin)
    y0 = max(0, int(min(ys)) - margin)
    x1 = min(WORK_W, int(max(xs)) + margin)
    y1 = min(WORK_H, int(max(ys)) + margin)
    return [x0, y0, x1 - x0, y1 - y0]


def scale_px_per_mm(quad, mm_corners):
    """由 A4 的像素边长与毫米边长估算"每毫米多少像素"（缩小图分辨率）。"""
    px_len = math.hypot(quad[1][0] - quad[0][0], quad[1][1] - quad[0][1])
    mm_len = math.hypot(mm_corners[1][0] - mm_corners[0][0],
                        mm_corners[1][1] - mm_corners[0][1])
    if mm_len <= 1e-6:
        return 1.0
    return px_len / mm_len


def run_pieces(cam, disp):
    """碎片识别模式：先用四角标定，再把碎片顶点换算成毫米并打印每片位姿。
    标定只在开机做一次（纸与相机不动）；碎片识别按 ANALYZE_INTERVAL_SEC 降频。"""
    frames = 0
    t_begin = time.time()
    quad = None
    px_to_mm = None
    min_edge_px = 0.0
    last_analyze = 0.0
    shapes = []
    lines = []
    while not app.need_exit():
        img = cam.read()
        work = build_work_image(img)
        now = time.time()

        if quad is None:
            if A4_QUAD_OVERRIDE:     # 用固定四角，不再依赖 A4 能不能被检出
                quad = [(float(px), float(py)) for px, py in A4_QUAD_OVERRIDE]
            else:
                quad = find_a4_quad(work)
            if quad is not None:
                mm_corners, orientation = mm_corners_for(quad)
                px_to_mm = vision_geo.compute_homography(quad, mm_corners)
                min_edge_px = MIN_EDGE_MM * scale_px_per_mm(quad, mm_corners)
                print("A4 标定完成：方向=%s 四角=%s | 每毫米 %.3f 像素，假边阈值 %.1f 像素" %
                      (orientation, quad, scale_px_per_mm(quad, mm_corners), min_edge_px))
        elif now - last_analyze >= ANALYZE_INTERVAL_SEC:
            last_analyze = now
            shapes = vision_pieces.find_piece_shapes(work, img, WORK_SCALE, WORK_W, WORK_H,
                                                     a4_roi(quad), min_edge_px)
            lines = [vision_pieces.format_piece_mm(
                index, [vision_geo.apply_homography(px_to_mm, x, y) for x, y in shape["verts"]])
                     for index, shape in enumerate(shapes)]

        if quad is not None:     # A4 外框用灰色细线画，表示毫米坐标系范围
            pixels = [(px * WORK_SCALE, py * WORK_SCALE) for px, py in quad]
            for i in range(4):
                x1, y1 = pixels[i]
                x2, y2 = pixels[(i + 1) % 4]
                img.draw_line(int(x1), int(y1), int(x2), int(y2), image.COLOR_GRAY)
        for index, shape in enumerate(shapes):
            color = PIECE_COLORS[index % len(PIECE_COLORS)]
            verts = shape["verts"]
            count = len(verts)
            for i in range(count):
                x1, y1 = verts[i]
                x2, y2 = verts[(i + 1) % count]
                img.draw_line(int(x1 * WORK_SCALE), int(y1 * WORK_SCALE),
                              int(x2 * WORK_SCALE), int(y2 * WORK_SCALE), color)
            cx, cy = int(shape["center"][0] * WORK_SCALE), int(shape["center"][1] * WORK_SCALE)
            img.draw_cross(cx, cy, color, 6)
            img.draw_string(cx + 8, cy + 8, str(index), color, scale=1)
        disp.show(img)

        frames += 1
        if now - t_begin >= STATUS_REPORT_SEC:
            if quad is None:
                print("fps %.1f | 等待 A4 标定：放好 A4，或把 a4_calib 打印的那行填进 A4_QUAD_OVERRIDE" %
                      (frames / (now - t_begin)))
            else:
                print("fps %.1f | pieces=%d" % (frames / (now - t_begin), len(shapes)))
                for text in lines:
                    print("   " + text)
                if not shapes:
                    count, biggest = vision_pieces.probe(work, a4_roi(quad))
                    print("   未检出：阈值内候选=%d 最大=%dpx（判据 %d~%dpx）"
                          "｜候选=0 查阈值/光照，候选>0 查尺寸判据"
                          % (count, biggest, PIECE_MIN_PIXELS, PIECE_MAX_PIXELS))
            frames = 0
            t_begin = now


def main():
    cam = camera.Camera(CAM_W, CAM_H)
    disp = display.Display()

    serial = open_uart() if USE_UART else None
    if serial is not None:
        serial.write_str("ready\r\n")   # 下位机收到这行说明物理链路已通

    vision_pieces.configure(THRESHOLDS=PIECE_THRESHOLDS, INVERT=PIECE_INVERT,
                            MIN_PIXELS=PIECE_MIN_PIXELS, MAX_PIXELS=PIECE_MAX_PIXELS)

    exposure, gain = setup_camera(cam)
    print("camera set: exposure=%s us, gain=%s | RUN_MODE=%s" % (exposure, gain, RUN_MODE))

    if RUN_MODE == "a4_calib":
        run_a4_calib(cam, disp)
    elif RUN_MODE == "pieces":
        run_pieces(cam, disp)
    else:
        print("未知 RUN_MODE:", RUN_MODE)


main()
