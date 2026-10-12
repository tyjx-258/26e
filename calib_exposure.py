# calib_exposure.py - 曝光/增益档位标定工具（现场一次性使用，不参与正式流程）
#
# 用途：按 SWEEP_POINTS 逐档试 (曝光微秒, 增益)，统计每档下最大亮区在 L50/L70/L90
#       三个亮度门槛的占比，并按"纸够亮、但没被拍成死白"给出推荐档位。
#       把 PICK 行里的两个数值填进 main.py 的 EXPOSURE_US 与 GAIN_VALUE 即完成固定。
# 运行方式：MaixVision 打开本工程文件夹 -> 单独运行本文件（不要运行完整项目）。
# 读数含义：L50 是纸成块的占比，L70 反映纸够不够亮，L90 反映有没有死白。

from maix import app, camera, display

# ---- 配置区 ----
CAM_W = 640
CAM_H = 480

SWEEP_POINTS = [                   # 每项为 (曝光微秒, 增益)
    (16000, 200),                  # 实测本机型增益量程到 1024，此处覆盖 200~1024
    (16000, 300),
    (16000, 400),
    (16000, 500),
    (16000, 700),
    (16000, 900),
    (16000, 1024),
]
SETTLE_FRAMES = 5                  # 换档后先空跑几帧，等设置真正生效
MEASURE_FRAMES = 5                 # 每档取多少帧做统计
DETAIL_LEVELS = [50, 70, 90]       # 统计这些亮度门槛的亮区占比
PICK_L90_MAX = 0.10                # 推荐档筛选条件：L90 不超过此值（纸没被拍成死白）
MIN_BLOB_PIXELS = 500              # 统计时忽略小于此像素数的亮区


def largest_blob_share(img, level):
    """返回指定亮度门槛下，最大连通区包围盒占整幅画面的比例。

    过曝时整幅连成一块（接近 100%），欠曝时几乎没有亮区（接近 0%）。
    """
    thresholds = [[level, 100, -128, 127, -128, 127]]
    blobs = img.find_blobs(thresholds, pixels_threshold=MIN_BLOB_PIXELS)
    best = 0
    for blob in blobs:
        area = blob[2] * blob[3]
        if area > best:
            best = area
    return best / float(CAM_W * CAM_H)


def measure_point(cam, disp):
    """在当前曝光与增益下采样若干帧，返回各亮度门槛亮区占比的中位数（抗单帧抖动）。"""
    samples = {level: [] for level in DETAIL_LEVELS}
    for _ in range(MEASURE_FRAMES):
        img = cam.read()
        disp.show(img)
        for level in DETAIL_LEVELS:
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


def sweep(cam, disp):
    """按 SWEEP_POINTS 逐档试参数，每轮结束打印一行 PICK 推荐结论。"""
    while not app.need_exit():
        rows = []
        for index, (exposure, gain) in enumerate(SWEEP_POINTS):
            try:
                cam.exposure(exposure)
                cam.gain(gain)
            except Exception as e:
                print("set exposure/gain failed:", e)

            for _ in range(SETTLE_FRAMES):
                disp.show(cam.read())

            shares = measure_point(cam, disp)
            rows.append((exposure, gain, shares))

            text = "sweep %d/%d: exp=%6d us gain=%4d" % (
                index + 1, len(SWEEP_POINTS), exposure, gain)
            for level in DETAIL_LEVELS:
                text += " | L%d=%3.0f%%" % (level, shares[level] * 100)
            print(text)

        best = pick_point(rows)
        if best is not None:
            exposure, gain, shares = best
            print("PICK exposure=%d us gain=%d | L50=%.0f%% L70=%.0f%% L90=%.0f%%" % (
                exposure, gain, shares[50] * 100, shares[70] * 100, shares[90] * 100))


def main():
    cam = camera.Camera(CAM_W, CAM_H)
    disp = display.Display()
    print("曝光标定：把 PICK 行的数值填进 main.py 的 EXPOSURE_US 与 GAIN_VALUE")
    sweep(cam, disp)


main()
