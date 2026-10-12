# main.py - MaixCAM Pro 视觉端：A4 标定与稳定检测
#
# 职责：开机阶段连续多帧确认 A4 位置后锁定；运行期用锁定框作为识别 ROI，
#       同时用红框显示当帧实时检测结果，便于判断检测是否漂移。
# 硬件假设：MaixCAM / MaixCAM Pro 本体（自带屏幕）即可运行，不需要下位机。
# 使用方式：MaixVision 打开本文件夹 -> 运行完整项目；未接 STM32 时保持 USE_UART = False。
# 验证目标：绿框（锁定结果）是否贴合纸边；红框（实时结果）是否与绿框重合。
#
# 设计取舍：A4 纸和相机都是固定的，每帧重新识别只会引入抖动，
#           所以纸张位置只在开机时标定一次，运行期不再依赖单帧检测结果。

from maix import app, camera, display, image, pinmap, uart, err
import time

# ---- 配置区：只改这里，不改下面逻辑 ----
USE_UART = False         # 未接 STM32 时保持 False；接上后再改 True
UART_DEV = "/dev/ttyS1"  # 不要用 /dev/ttyS0：它会输出启动日志，A16 被外部拉低还可能导致无法启动
UART_TX_PIN = "A19"
UART_RX_PIN = "A18"
UART_BAUD = 115200       # 必须与 STM32 侧保持一致

CAM_W = 640              # 验证阶段用低分辨率保帧率，定好安装高度后再提高
CAM_H = 480              # 改分辨率后必须重新标定像素到毫米的换算关系

# A4 亮度阈值（LAB 颜色空间：L 0~100，A/B -128~127），按"白纸 + 深色背景"给起点。
# 识别不到就调小 A4_L_MIN，把台面也框进来就调大。
A4_L_MIN = 60
A4_L_MAX = 100
A4_A_MIN, A4_A_MAX = -30, 30
A4_B_MIN, A4_B_MAX = -30, 30
A4_THRESHOLDS = [[A4_L_MIN, A4_L_MAX, A4_A_MIN, A4_A_MAX, A4_B_MIN, A4_B_MAX]]

A4_BLOB_MIN_PIXELS = 500                  # 低于此像素数的亮区在取块阶段直接丢掉
A4_RATIO_MIN, A4_RATIO_MAX = 0.45, 0.95   # 竖向 A4 宽高比约 0.71，放宽以容忍轻微倾斜
A4_MIN_AREA_RATIO = 0.10                  # 候选框面积至少占画面比例，挡掉"整片台面"式误检

CONFIRM_FRAMES = 8        # 连续多少帧结果一致才认为标定成功
CENTER_TOL_PX = 15        # 相邻帧中心允许偏移（像素）
SIZE_TOL_RATIO = 0.15     # 相邻帧宽高允许变化比例

MONITOR_LIVE = True       # 运行期是否继续做实时检测（只用于观察，不参与决策）
STATUS_REPORT_SEC = 1.0   # 诊断信息打印间隔，避免刷屏拖低帧率


def open_uart():
    """按配置打开与下位机的串口；失败返回 None，让预览功能继续可用。"""
    err.check_raise(pinmap.set_pin_function(UART_TX_PIN, "UART1_TX"), "set UART1 TX failed")
    err.check_raise(pinmap.set_pin_function(UART_RX_PIN, "UART1_RX"), "set UART1 RX failed")
    try:
        return uart.UART(UART_DEV, UART_BAUD)
    except Exception as e:
        print("uart open failed:", e)
        return None


def find_best_a4(img):
    """返回画面中最像 A4 的候选框 (x, y, w, h)，没有合格候选时返回 None。

    先用亮度阈值取连通区，再用宽高比和面积占比过滤：
    宽高比挡掉细长反光和碎片本身，面积占比挡掉"整片台面"这种过大区域。
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
    """判断两个框是否可视为同一个目标：中心偏移和尺寸变化都在容差内。"""
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
    """生成每秒一条的诊断信息：看界面上是框丢了还是框在跳。"""
    locked_text = "none" if locked is None else "%d,%d,%d,%d" % locked
    live_text = "none" if live is None else "%d,%d,%d,%d" % live
    return "fps %.1f | state=%s stable=%d | locked=%s | live=%s" % (
        fps, state, stable, locked_text, live_text)


def main():
    cam = camera.Camera(CAM_W, CAM_H)
    disp = display.Display()

    serial = open_uart() if USE_UART else None
    if serial is not None:
        serial.write_str("ready\r\n")   # 下位机或串口助手收到这行，说明物理链路已通

    state = "calib"     # calib: 正在标定纸张位置；run: 已锁定
    stable = 0
    last = None
    locked = None

    frames = 0
    t_begin = time.time()
    while not app.need_exit():
        img = cam.read()
        live = find_best_a4(img) if MONITOR_LIVE or state == "calib" else None

        if state == "calib":
            # 只有连续多帧结果一致才锁定，避免用一帧抖动结果定下整场 ROI
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
