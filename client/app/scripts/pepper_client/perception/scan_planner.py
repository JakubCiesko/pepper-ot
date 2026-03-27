import math


def planned_yaws_radians(config):
    yaws_deg = config["capture"].get("scan_yaws_deg") or [-35, 0, 35]
    return [math.radians(float(value)) for value in yaws_deg]


def scan_pitch(config):
    return float(config["capture"].get("scan_head_pitch", -0.1))
