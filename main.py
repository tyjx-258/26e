# main.py - MaixCAM Pro 视觉端最小验证 Demo
#
# 职责：相机取图 -> 屏幕预览 -> 每秒打印帧率 -> 可选向 STM32 发一行握手字
# 硬件假设：MaixCAM Pro + 官方屏幕；自定义串口用 UART1，A19(TX)/A18(RX)，设备 /dev/ttyS1
# 使用方式：MaixVision 打开本文件夹并运行完整项目；未接 STM32 时把 USE_UART 改为 False
# 验证目标：视野是否覆盖整张纵向 A4 纸；曝光是否稳定；与下位机的物理串口是否通
#
# 本文件只做链路验证，不做碎片识别；识别与标定代码等安装高度确定后再加。

from maix import app, camera, display, image, pinmap, uart, err
import time

# ---- 配置区：只改这里，不改下面逻辑 ----
USE_UART = True          # 只调镜头/曝光时置 False，避免引脚复用影响其它功能
UART_DEV = "/dev/ttyS1"  # 不要用 /dev/ttyS0：它会输出启动日志，且 A16 被外部拉低可能导致无法启动
UART_TX_PIN = "A19"
UART_RX_PIN = "A18"
UART_BAUD = 115200       # 必须与 STM32 侧保持一致

CAM_W = 640              # 验证阶段用低分辨率保帧率，定好安装高度后再提高
CAM_H = 480              # 改分辨率后必须重新标定像素到毫米的换算关系

FPS_REPORT_SEC = 1.0


def open_uart():
    """配置 UART1 引脚并打开串口；失败时返回 None，让预览功能继续可用。"""
    err.check_raise(pinmap.set_pin_function(UART_TX_PIN, "UART1_TX"), "set UART1 TX failed")
    err.check_raise(pinmap.set_pin_function(UART_RX_PIN, "UART1_RX"), "set UART1 RX failed")
    try:
        return uart.UART(UART_DEV, UART_BAUD)
    except Exception as e:
        print("uart open failed:", e)
        return None


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

        # 左上角固定绿框：确认图像坐标原点方向，也便于观察静止物体是否抖动
        img.draw_rect(20, 20, 100, 60, image.COLOR_GREEN)
        disp.show(img)

        frames += 1
        t_now = time.time()
        if t_now - t_begin >= FPS_REPORT_SEC:
            print("fps: %.1f" % (frames / (t_now - t_begin)))
            frames = 0
            t_begin = t_now


main()
