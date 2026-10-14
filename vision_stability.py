# vision_stability.py - 拼接求解前的稳定性门
#
# 作用：只有连续多帧都识别到指定片数，且各片质心移动很小时，才允许触发拼接。
# 这样可以避免碎片还在摆放、遮挡、漏检时反复跑自动求解，提升摆件阶段帧率。

STABLE_PIECE_COUNT = 4
STABLE_FRAMES = 3
MOVE_TOL_PX = 5.0

_stable_frames = 0
_last_centers = None


def update(shapes):
    """更新稳定性计数；稳定返回 True，否则返回 False。"""
    global _stable_frames, _last_centers
    if len(shapes) != STABLE_PIECE_COUNT:
        _stable_frames = 0
        _last_centers = None
        return False

    centers = [(shape["center"][0], shape["center"][1]) for shape in shapes]
    if _last_centers is None or len(_last_centers) != len(centers):
        _stable_frames = 1
    else:
        max_move = 0.0
        for (x1, y1), (x2, y2) in zip(_last_centers, centers):
            max_move = max(max_move, ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5)
        _stable_frames = _stable_frames + 1 if max_move <= MOVE_TOL_PX else 1
    _last_centers = centers
    return _stable_frames >= STABLE_FRAMES
