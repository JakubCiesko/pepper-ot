import math


def planned_yaws_radians(config):
    yaws_deg = config["capture"].get("scan_yaws_deg") or [-35, 0, 35]
    return [math.radians(float(value)) for value in yaws_deg]


def scan_pitch(config):
    return float(config["capture"].get("scan_head_pitch", -0.1))


def scan_mode(config):
    panorama = config.get("panorama", {})
    if not panorama.get("enabled", True):
        return "sequential_detect"
    mode = str(panorama.get("mode") or "panorama_detect").strip().lower()
    if mode not in ("panorama_detect", "sequential_detect"):
        return "panorama_detect"
    return mode


def stick_together(config):
    return bool(config.get("panorama", {}).get("stick_together", True))


def summary_after_scan(config):
    panorama = config.get("panorama", {})
    if "summary_after_scan" in panorama:
        return bool(panorama.get("summary_after_scan"))
    return bool(config.get("behavior", {}).get("allow_scan_summary_chat", True))


def memory_render_limit(config):
    tablet_cfg = config.get("tablet", {})
    panorama_cfg = config.get("panorama", {})
    candidate = tablet_cfg.get("memory_render_limit", panorama_cfg.get("render_limit", 5))
    try:
        return max(1, int(candidate))
    except Exception:
        return 5
