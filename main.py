# main.py - MaixCAM Pro 视觉端主程序
#
# 职责：固定相机曝光/增益/白平衡 -> 开机连续多帧确认 A4 位置并锁定 -> 运行期以锁定框作为识别 ROI。
# 硬件假设：MaixCAM / MaixCAM Pro 本体（自带屏幕）；未接下位机时把 USE_UART 置 False。
# 运行方式：MaixVision 打开本工程文件夹 -> 运行完整项目。
# 配套工具：曝光/增益档位标定见同目录 calib_exposure.py，单独运行，不参与正式流程。
#
# API 依据（Sipeed 官方 MaixPy 文档）：
#   cam.exposure(value)：单位微秒；设置后进入手动曝光，不传参返回当前值
#   cam.gain(value)：自定义增益仅在手动曝光模式下生效
#   cam.set_wb_gain([r, gr, gb, b])：范围 0~1.0，调用即禁用自动白平衡

from maix import app, camera, display, image, pinmap, uart, err
import time

# ---- 串口（对接下位机）----
USE_UART = False         # 未接 STM32 时保持 False；接上后改 True
UART_DEV = "/dev/ttyS1"  # 不要用 /dev/ttyS0：它会输出启动日志，A16 被外部拉低还可能导致无法启动
UART_TX_PIN = "A19"
UART_RX_PIN = "A18"
UART_BAUD = 115200

# ---- 相机固定参数（数值由 calib_exposure.py 标定得到）----
CAM_W = 640
CAM_H = 480
EXPOSURE_US = 16000              # 实测该光照下 16000us + 增益 500 画面正常：白纸白、黑垫黑
GAIN_VALUE = 500
WARMUP_SEC = 0.5                 # 空跑少量帧让画面稳定，再锁白平衡
LOCK_WHITE_BALANCE = True
WB_GAIN_FALLBACK = [0.134, 0.0625, 0.0625, 0.1239]   # 官方给出的 MaixCAM 初始白平衡增益

# ---- A4 检测参数 ----
A4_L_MIN = 50                    # 白纸亮度下限；实测纸约 L50~85、黑垫低于 L50
A4_L_MAX = 100
A4_A_MIN, A4_A_MAX = -60, 60
A4_B_MIN, A4_B_MAX = -60, 60
A4_THRESHOLDS = [[A4_L_MIN, A4_L_MAX, A4_A_MIN, A4_A_MAX, A4_B_MIN, A4_B_MAX]]

A4_BLOB_MIN_PIXELS = 500         # 小于此像素数的亮区当噪声
A4_MIN_AREA_RATIO = 0.08         # 包围盒面积占画面比例下限
A4_RATIO_MIN, A4_RATIO_MAX = 0.4, 2.0   # 只用于挡细长反光，不作为主要判据

CONFIRM_FRAMES = 8               # 连续多少帧结果一致才锁定纸张位置
CENTER_TOL_PX = 20               # 相邻帧中心允许偏移（像素）
SIZE_TOL_RATIO = 0.15            # 相邻帧宽高允许变化比例

# ---- 运行开关 ----
VERBOSE = False                  # True：每秒打印检测状态，现场调试用
STATUS_REPORT_SEC = 1.0
FULL_FRAME_WARN_RATIO = 0.85     # 实时框占比超过此值时告警，提示曝光或阈值异常


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


def lock_camera(cam):
    """固定曝光与增益，避免自动曝光随时间漂移导致整幅画面过曝。

    任一步失败都降级为自动模式继续运行，只打印提示。返回 (曝光微秒, 增益) 供记录。
    """
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


def find_best_a4(img):
    """返回画面中最像 A4 的候选框 (x, y, w, h)，没有合格候选时返回 None。

    先用亮度阈值取连通区，再用面积占比和宽高比粗过滤。坐标为像素，原点在画面左上角。
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


def main():
    cam = camera.Camera(CAM_W, CAM_H)
    disp = display.Display()

    serial = open_uart() if USE_UART else None
    if serial is not None:
        serial.write_str("ready\r\n")   # 下位机收到这行说明物理链路已通

    exposure, gain = lock_camera(cam)
    print("camera locked: exposure=%s us, gain=%s" % (exposure, gain))

    state = "calib"     # calib：正在确认纸张位置；run：已锁定
    stable = 0
    last = None
    locked = None

    frames = 0
    t_begin = time.time()
    while not app.need_exit():
        img = cam.read()
        live = find_best_a4(img)

        if state == "calib":
            # 连续多帧一致才锁定，避免用一帧抖动结果定下整场 ROI
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
            share = 0.0 if live is None else (live[2] * live[3]) / float(CAM_W * CAM_H)
            if VERBOSE:
                print("fps %.1f | state=%s stable=%d | locked=%s | live=%s (%.0f%%)" % (
                    frames / (t_now - t_begin), state, stable, locked, live, share * 100))
            elif share >= FULL_FRAME_WARN_RATIO:
                print("WARN 整幅画面都是亮区，检查曝光与阈值")
            frames = 0
            t_begin = t_now


main()
