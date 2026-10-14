# vision_mode.py - 视觉端模式切换（模板拼接 / 自动拼接）
#
# 下位机通过 UART 发一行命令切换：
#   M,0 或 MODE,T -> 模板拼接
#   M,1 或 MODE,A -> 自动拼接
# MaixCAM 收到后回一行 MODE,<name> 作为确认。

MODE_TEMPLATE = "template"
MODE_AUTO = "auto"
DEFAULT_MODE = MODE_AUTO       # 下位机未接入时，先默认自动拼接

_mode = DEFAULT_MODE
_buffer = ""


def get_mode():
    """返回当前模式名："template" 或 "auto"。"""
    return _mode


def poll(serial):
    """轮询 UART 模式命令；serial 为 None 时保持当前模式。"""
    global _mode, _buffer
    if serial is None:
        return
    try:
        data = serial.read()
    except Exception:
        return
    if not data:
        return
    text = data.decode("utf-8", "ignore") if isinstance(data, bytes) else str(data)
    _buffer += text
    while "\n" in _buffer:
        line, _buffer = _buffer.split("\n", 1)
        command = line.strip().upper()
        if command in ("M,0", "MODE,T", "TEMPLATE"):
            _mode = MODE_TEMPLATE
        elif command in ("M,1", "MODE,A", "AUTO"):
            _mode = MODE_AUTO
        else:
            continue
        try:
            serial.write_str("MODE,%s\n" % _mode)
        except Exception:
            pass
