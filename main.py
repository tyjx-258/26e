# main.py - MaixCAM Pro 视觉端：曝光标定 + A4 检测与标定锁定
#
# 职责：开机先确定相机曝光/增益（消除自动曝光漂移与过曝），再连续多帧确认 A4 位置并锁定；
#       运行期用锁定框作为识别 ROI，红框显示当帧实时检测结果。
# 硬件假设：MaixCAM / MaixCAM Pro 本体（自带屏幕）即可运行，不需要下位机。
# 使用方式：MaixVision 打开本文件夹 -> 运行完整项目；未接 STM32 时保持 USE_UART = False。
#
# API 依据（Sipeed 官方文档，MaixPy）：
#   cam.exposure(value) 单位微秒，设置后进入手动曝光；不传参返回当前值
#   cam.gain(value)     自定义增益仅在手动曝光模式下生效
#   cam.set_wb_gain([r, gr, gb, b])  范围 0~1.0，调用即禁用自动白平衡
#   exp_mode / awb_mode 已被官方标记 deprecated，本文件不使用
#
# 用法：先用 CALIBRATE_EXPOSURE = True 跑一次，按画面挑出合适曝光值；
#       再把该值填进 EXPOSURE_US 并将 CALIBRATE_EXPOSURE 改回 False，进入正常工作流程。

from maix import app, camera, display, image, pinmap, uart, err
import time

# ---- 配置区：只改这里，不改下面逻辑 ----
USE_UART = False         # 未接 STM32 时保持 False；接上后再改 True
UART_DEV = "/dev/ttyS1"  # 不要用 /dev/ttyS0：它会输出启动日志，A16 被外部拉低还可能导致无法启动
UART_TX_PIN = "A19"
UART_RX_PIN = "A18"
UART_BAUD = 115200

CAM_W = 640
CAM_H = 480

# ---- 曝光/增益标定模式 ----
CALIBRATE_EXPOSURE = False         # True：扫描曝光与增益并给出推荐档；已定好数值，置 False 进入正常流程
SWEEP_POINTS = [                   # 每项为 (曝光微秒, 增益)
    (16000, 500),                  # 单点保持：先肉眼确认这一组参数下的画面是否正常
]
SETTLE_FRAMES = 5                  # 换档后先空跑几帧，等设置真正生效
MEASURE_FRAMES = 5                 # 每档取多少帧做统计
SWEEP_DETAIL_LEVELS = [50, 70, 90] # 统计这些亮度档的亮区占比
PICK_L90_MAX = 0.10                # 推荐档筛选条件：L90 不超过这么多（纸没被拍成死白）

# ---- 相机固定参数（CALIBRATE_EXPOSURE = False 时生效）----
FIX_CAMERA = True
EXPOSURE_US = 16000                # 标定得到：16000us + 增益 500 时曝光正常（白纸白、黑垫黑）
GAIN_VALUE = 500                   # 0：沿用当前增益；>0：固定为该增益
LOCK_WHITE_BALANCE = True
WB_GAIN_FALLBACK = [0.134, 0.0625, 0.0625, 0.1239]   # 官方给出的 MaixCAM 初始白平衡增益
WARMUP_SEC = 0.5                   # 已给固定曝光值，只需少量帧让画面稳定

# ---- A4 检测参数 ----
A4_L_MIN = 50             # 曝光标定好之后再按实际画面复核这个阈值
A4_L_MAX = 100
A4_A_MIN, A4_A_MAX = -60, 60
A4_B_MIN, A4_B_MAX = -60, 60
A4_THRESHOLDS = [[A4_L_MIN, A4_L_MAX, A4_A_MIN, A4_A_MAX, A4_B_MIN, A4_B_MAX]]

A4_BLOB_MIN_PIXELS = 500
A4_MIN_AREA_RATIO = 0.08
A4_RATIO_MIN, A4_RATIO_MAX = 0.4, 2.0

CONFIRM_FRAMES = 8
CENTER_TOL_PX = 20
SIZE_TOL_RATIO = 0.15

MONITOR_LIVE = True
STATUS_REPORT_SEC = 1.0
FULL_FRAME_WARN_RATIO = 0.85


def open_uart():
    """按配置打开与下位机的串口；失败返回 None，让预览功能继续可用。"""
    err.check_raise(pinmap.set_pin_function(UART_TX_PIN, "UART1_TX"), "set UART1 TX failed")
    err.check_raise(pinmap.set_pin_function(UART_RX_PIN, "UART1_RX"), "set UART1 RX failed")
    try:
        return uart.UART(UART_DEV, UART_BAUD)
    except Exception as e:
        print("uart open failed:", e)
        return None


def read_camera_params(cam):
    """读取当前曝光与增益，仅用于打印记录；取不到时返回 (None, None)。"""
    try:
        return cam.exposure(), cam.gain()
    except Exception as e:
        print("read camera params failed:", e)
        return None, None


def largest_blob_share(img, l_min):
    """返回指定亮度下限下，最大连通区包围盒占整幅画面的比例。

    用于比较不同曝光档位：过曝时整幅连成一块（接近 100%），
    欠曝时几乎没有亮区（接近 0%）；只有成块且占比适中的亮区才说明曝光合适。
    """
    thresholds = [[l_min, 100, -128, 127, -128, 127]]
    blobs = img.find_blobs(thresholds, pixels_threshold=A4_BLOB_MIN_PIXELS)
    best = 0
    for blob in blobs:
        area = blob[2] * blob[3]
        if area > best:
            best = area
    return best / float(CAM_W * CAM_H)


def measure_point(cam, disp):
    """在当前曝光与增益下采样若干帧，返回各亮度档亮区占比的中位数（抗单帧抖动）。"""
    samples = {level: [] for level in SWEEP_DETAIL_LEVELS}
    for _ in range(MEASURE_FRAMES):
        img = cam.read()
        disp.show(img)
        for level in SWEEP_DETAIL_LEVELS:
            samples[level].append(largest_blob_share(img, level))

    shares = {}
    for level, values in samples.items():
        values.sort()
        shares[level] = values[len(values) // 2]
    return shares


def pick_point(rows):
    """从一轮扫描结果里挑推荐档：在"没有死白"的档位中选纸最亮的那个。

    rows 为 [(exposure, gain, shares_dict), ...]。
    若没有任何档位满足 L90 上限，则退化为 L70 占比最大的一档，保证仍有结论。
    """
    if not rows:
        return None
    safe = [row for row in rows if row[2][90] <= PICK_L90_MAX]
    return max(safe if safe else rows, key=lambda row: row[2][70])


def calibrate_exposure(cam, disp):
    """按 SWEEP_POINTS 逐档试曝光与增益，并给出一行推荐结论。

    合适档位特征：纸够亮（L70 有份额）、但没被拍成死白（L90 接近 0）。
    每轮结束打印 PICK 行，把那一行贴回即可确定 EXPOSURE_US 与 GAIN_VALUE。
    """
    while not app.need_exit():
        rows = []
        for index, (exposure, gain) in enumerate(SWEEP_POINTS):
            try:
                cam.exposure(exposure)
                cam.gain(gain)
            except Exception as e:
                print("set exposure/gain failed:", e)

            for _ in range(SETTLE_FRAMES):    # 换档后先空跑几帧，等设置真正生效
                disp.show(cam.read())

            shares = measure_point(cam, disp)
            rows.append((exposure, gain, shares))

            text = "sweep %d/%d: exp=%6d us gain=%4d" % (
                index + 1, len(SWEEP_POINTS), exposure, gain)
            for level in SWEEP_DETAIL_LEVELS:
                text += " | L%d=%3.0f%%" % (level, shares[level] * 100)
            print(text)

        best = pick_point(rows)
        if best is not None:
            exposure, gain, shares = best
            print("PICK exposure=%d us gain=%d | L50=%.0f%% L70=%.0f%% L90=%.0f%%" % (
                exposure, gain, shares[50] * 100, shares[70] * 100, shares[90] * 100))


def lock_white_balance(cam):
    """锁定白平衡。接口不可用时只提示并保持自动白平衡，不影响主流程。"""
    try:
        gains = cam.set_wb_gain()
        gains = list(gains) if gains else list(WB_GAIN_FALLBACK)
        cam.set_wb_gain(gains)
        print("white balance locked:", [round(float(g), 4) for g in gains])
    except Exception as e:
        print("white balance lock skipped:", e)


def lock_camera(cam):
    """固定曝光与增益，避免自动调节随时间漂移；任一步失败都降级为自动模式继续运行。

    返回 (曝光微秒, 增益)，取不到时返回 (None, None)，仅用于打印记录。
    """
    t0 = time.time()
    while not app.need_exit() and time.time() - t0 < WARMUP_SEC:
        cam.read()

    try:
        exposure = cam.exposure() if EXPOSURE_US < 0 else EXPOSURE_US
        if exposure > 0:
            cam.exposure(exposure)
        if GAIN_VALUE > 0:
            cam.gain(GAIN_VALUE)
    except Exception as e:
        print("exposure/gain lock skipped:", e)

    if LOCK_WHITE_BALANCE:
        lock_white_balance(cam)

    return read_camera_params(cam)


def find_best_a4(img):
    """返回画面中最像 A4 的候选框 (x, y, w, h)，没有合格候选时返回 None。

    先用亮度阈值取连通区，再用面积占比和宽高比粗过滤。
    坐标为像素，原点在画面左上角。
    """
    blobs = img.find_blobs(A4_THRESHOLDS, pixels_threshold=A4_BLOB_MIN_PIXELS)
    frame_area = CAM_W * CAM_H
    best = None
    best_area = 0
    for blob in blobs:
        x, y, w, h = blob[0], blob[1], blob[2], blob[3]
        if h <= 0:
            continue
        ratio = w / h
        if ratio < A4_RATIO_MIN or ratio > A4_RATIO_MAX:
            continue
        area = w * h
        if area < frame_area * A4_MIN_AREA_RATIO:
            continue
        if area > best_area:
            best_area = area
            best = (x, y, w, h)
    return best


def is_same_target(a, b):
    """判断两个框是否可视为同一目标：中心偏移和尺寸变化都在容差内。"""
    if a is None or b is None:
        return False
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    if abs((ax + aw / 2) - (bx + bw / 2)) > CENTER_TOL_PX:
        return False
    if abs((ay + ah / 2) - (by + bh / 2)) > CENTER_TOL_PX:
        return False
    if abs(aw - bw) > aw * SIZE_TOL_RATIO:
        return False
    if abs(ah - bh) > ah * SIZE_TOL_RATIO:
        return False
    return True


def format_status(state, stable, locked, live, fps):
    """生成每秒一条的诊断信息，用来判断是框丢了、框在跳，还是曝光漂了。"""
    locked_text = "none" if locked is None else "%d,%d,%d,%d" % locked
    live_text = "none" if live is None else "%d,%d,%d,%d" % live
    share = 0.0 if live is None else (live[2] * live[3]) / float(CAM_W * CAM_H)
    text = "fps %.1f | state=%s stable=%d | locked=%s | live=%s (%.0f%%)" % (
        fps, state, stable, locked_text, live_text, share * 100)
    if locked is not None and live is not None:
        text += " | d=(%d,%d)" % (live[0] - locked[0], live[1] - locked[1])
    if share >= FULL_FRAME_WARN_RATIO:
        text += " | WARN 整幅画面都是亮区，检查曝光与阈值"
    return text


def main():
    cam = camera.Camera(CAM_W, CAM_H)
    disp = display.Display()

    exposure, gain = read_camera_params(cam)
    print("camera now: exposure=%s us, gain=%s" % (exposure, gain))

    if CALIBRATE_EXPOSURE:
        print("曝光标定模式：每档约 2 秒并循环，把控制台整段贴回用于选档")
        calibrate_exposure(cam, disp)
        return

    serial = open_uart() if USE_UART else None
    if serial is not None:
        serial.write_str("ready\r\n")   # 下位机或串口助手收到这行，说明物理链路已通

    if FIX_CAMERA:
        exposure, gain = lock_camera(cam)
        print("camera locked: exposure=%s us, gain=%s" % (exposure, gain))

    state = "calib"     # calib: 正在标定纸张位置；run: 已锁定
    stable = 0
    last = None
    locked = None

    frames = 0
    t_begin = time.time()
    while not app.need_exit():
        img = cam.read()
        live = find_best_a4(img) if (MONITOR_LIVE or state == "calib") else None

        if state == "calib":
            stable = stable + 1 if is_same_target(last, live) else (1 if live else 0)
            last = live
            if stable >= CONFIRM_FRAMES:
                locked = live
                state = "run"
                print("A4 locked: x=%d y=%d w=%d h=%d" % locked)

        if live is not None:
            img.draw_rect(live[0], live[1], live[2], live[3], image.COLOR_RED)
        if locked is not None:
            img.draw_rect(locked[0], locked[1], locked[2], locked[3], image.COLOR_GREEN)
        disp.show(img)

        frames += 1
        t_now = time.time()
        if t_now - t_begin >= STATUS_REPORT_SEC:
            print(format_status(state, stable, locked, live, frames / (t_now - t_begin)))
            frames = 0
            t_begin = t_now


main()
