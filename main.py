# main.py - MaixCAM Pro 视觉端最小验证 Demo
#
# 职责：相机取图 -> 检测 A4 纸张 -> 屏幕叠加检测框 -> 控制台打印检测结果与帧率
# 硬件假设：MaixCAM / MaixCAM Pro 本体（自带屏幕）即可运行，不需要下位机
# 使用方式：MaixVision 打开本文件夹并选择"运行完整项目"；未接 STM32 时把 USE_UART 改为 False
# 验证目标：视野是否覆盖整张纵向 A4 纸；A4 检测框是否稳定贴合纸边；曝光是否可用
#
# 本文件只做相机与 A4 检测验证，不做碎片轮廓识别；识别与标定代码等安装高度确定后再加。

from maix import app, camera, display, image, pinmap, uart, err
import time

# ---- 配置区：只改这里，不改下面逻辑 ----
USE_UART = False         # 现在只有 MaixCAM 本体，先保持 False；接上下位机后再改 True
UART_DEV = "/dev/ttyS1"  # 不要用 /dev/ttyS0：它会输出启动日志，A16 被外部拉低还可能导致无法启动
UART_TX_PIN = "A19"
UART_RX_PIN = "A18"
UART_BAUD = 115200       # 必须与 STM32 侧保持一致

CAM_W = 640              # 验证阶段用低分辨率保帧率，定好安装高度后再提高
CAM_H = 480              # 改分辨率后必须重新标定像素到毫米的换算关系

# A4 检测阈值，LAB 颜色空间（L:0~100，A/B:-128~127）
# 默认按"白色纸 + 深色台面"给起点：亮度高、色彩中性。
# 识别不到就调小 A4_L_MIN（放宽），把台面也框进来就调大（收紧）。
A4_L_MIN = 60
A4_L_MAX = 100
A4_A_MIN, A4_A_MAX = -30, 30
A4_B_MIN, A4_B_MAX = -30, 30
A4_THRESHOLDS = [[A4_L_MIN, A4_L_MAX, A4_A_MIN, A4_A_MAX, A4_B_MIN, A4_B_MAX]]

A4_MIN_PIXELS = 2000     # 小于此像素数的亮区当噪声丢掉
A4_MIN_WIDTH = 120       # A4 在画面里的最小宽高，用来挡掉小块反光
A4_MIN_HEIGHT = 120

STATUS_REPORT_SEC = 1.0  # 控制台状态打印间隔，避免刷屏拖低帧率


def open_uart():
    """按配置打开与下位机的串口；失败返回 None，让预览功能继续可用。"""
    err.check_raise(pinmap.set_pin_function(UART_TX_PIN, "UART1_TX"), "set UART1 TX failed")
    err.check_raise(pinmap.set_pin_function(UART_RX_PIN, "UART1_RX"), "set UART1 RX failed")
    try:
        return uart.UART(UART_DEV, UART_BAUD)
    except Exception as e:
        print("uart open failed:", e)
        return None


def find_a4(img):
    """在整幅画面里找最像 A4 的亮区，返回 (x, y, w, h)，找不到返回 None。

    只用"亮度阈值 + 尺寸下限"筛选：白纸在深色台面上是画面里最大的亮块。
    返回坐标为像素，原点在画面左上角。
    """
    blobs = img.find_blobs(A4_THRESHOLDS, pixels_threshold=A4_MIN_PIXELS)
    best = None
    best_area = 0
    for blob in blobs:
        x, y, w, h = blob[0], blob[1], blob[2], blob[3]
        if w < A4_MIN_WIDTH or h < A4_MIN_HEIGHT:
            continue
        if w * h > best_area:
            best_area = w * h
            best = (x, y, w, h)
    return best


def main():
    cam = camera.Camera(CAM_W, CAM_H)
    disp = display.Display()

    serial = open_uart() if USE_UART else None
    if serial is not None:
        serial.write_str("ready\r\n")   # 下位机或串口助手收到这行，说明物理链路已通

    frames = 0
    t_begin = time.time()
    while not app.need_exit():
        img = cam.read()
        a4 = find_a4(img)

        if a4 is not None:
            x, y, w, h = a4
            # 绿框应贴合纸张四边：框比纸小说明阈值太紧，框住整片台面说明阈值太松
            img.draw_rect(x, y, w, h, image.COLOR_GREEN)
        disp.show(img)

        frames += 1
        t_now = time.time()
        if t_now - t_begin >= STATUS_REPORT_SEC:
            fps = frames / (t_now - t_begin)
            if a4 is None:
                print("fps %.1f | A4: NOT found" % fps)
            else:
                x, y, w, h = a4
                print("fps %.1f | A4: x=%d y=%d w=%d h=%d center=(%d,%d)"
                      % (fps, x, y, w, h, x + w // 2, y + h // 2))
            frames = 0
            t_begin = t_now


main()
