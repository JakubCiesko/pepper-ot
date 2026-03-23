import copy
import json
import os


DEFAULT_CONFIG = {
    "app": {
        "app_id": "com.aldebaran.PepperGroundedClient",
        "service_name": "PepperGroundedClient",
    },
    "server": {
        "base_url": "http://127.0.0.1:8000",
        "caption_path": "/api/v1/caption",
        "detect_path": "/api/v1/detect",
        "chat_path": "/api/v1/chat",
        "config_patch_path": "/api/v1/config",
        "dashboard_url": "http://127.0.0.1:8000/dashboard",
        "publish": True,
        "verify_tls": True,
        "caption_timeout_seconds": 20,
        "detect_timeout_seconds": 60,
        "chat_timeout_seconds": 20,
        "config_timeout_seconds": 20,
    },
    "capture": {
        "camera_id": 0,
        "resolution": 2,
        "color_space": 11,
        "fps": 10,
        "jpeg_quality": 90,
        "scan_yaws_deg": [-35, 0, 35],
        "scan_head_pitch": -0.1,
        "head_move_speed": 0.15,
        "settle_seconds": 0.6,
        "refresh_ttl_seconds": 25,
        "frame_prefix": "pepper_frame",
        "scan_prefix": "pepper_scan",
        "scan_summary_query_en": "Briefly describe what you can see now using the current visual memory.",
        "scan_summary_query_cs": "Strucne popis co ted vidis podle aktualni vizualni pameti.",
    },
    "behavior": {
        "caption_run_detect": True,
        "caption_retry_on_timeout": True,
        "auto_refresh_before_chat": True,
        "allow_scan_summary_chat": True,
        "show_dashboard_on_start": False,
        "show_dashboard_during_scan": False,
        "speak_acknowledgements": True,
        "auto_restore_head_pose": True,
        "max_query_chars": 500,
    },
    "social": {
        "enable_people_perception": True,
        "enable_face_detection": True,
        "enable_face_characteristics": True,
        "enable_gaze_analysis": True,
        "enable_engagement_zones": True,
        "enable_sitting_detection": True,
        "enable_waving_detection": True,
        "enable_sonar": True,
        "face_match_max_angle_rad": 0.35,
        "expression_labels": ["neutral", "happy", "surprised", "angry", "sad"],
    },
    "language": {
        "default_dialog_language": "en",
        "output_language_mode": "default",
    },
}

VALID_OUTPUT_LANGUAGES = ("default", "english", "czech")


def _deep_merge(base, override):
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value
    return base


def load_config(path, logger=None):
    config = copy.deepcopy(DEFAULT_CONFIG)
    if path and os.path.exists(path):
        try:
            with open(path, "r") as handle:
                payload = json.load(handle)
            if isinstance(payload, dict):
                _deep_merge(config, payload)
        except Exception as exc:
            if logger is not None:
                logger.warning("Failed to load client config %s: %s", path, exc)
    elif logger is not None:
        logger.info("Client config file not found, using defaults: %s", path)
    normalize_config(config)
    config["_config_path"] = path
    return config


def normalize_config(config):
    base_url = str(config["server"].get("base_url") or "").strip()
    if base_url.endswith("/"):
        base_url = base_url[:-1]
    config["server"]["base_url"] = base_url
    dashboard_url = str(config["server"].get("dashboard_url") or "").strip()
    config["server"]["dashboard_url"] = dashboard_url

    scan_yaws = config["capture"].get("scan_yaws_deg") or [-35, 0, 35]
    if not isinstance(scan_yaws, list) or not scan_yaws:
        scan_yaws = [-35, 0, 35]
    config["capture"]["scan_yaws_deg"] = scan_yaws

    mode = normalize_output_language(config["language"].get("output_language_mode"))
    config["language"]["output_language_mode"] = mode
    return config


def normalize_output_language(mode):
    normalized = str(mode or "default").strip().lower()
    if normalized not in VALID_OUTPUT_LANGUAGES:
        return "default"
    return normalized


def save_config(config, logger=None):
    path = config.get("_config_path")
    if not path:
        return False
    payload = copy.deepcopy(config)
    payload.pop("_config_path", None)
    try:
        with open(path, "w") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        if logger is not None:
            logger.info("Saved client config to %s", path)
        return True
    except Exception as exc:
        if logger is not None:
            logger.warning("Failed to save client config to %s: %s", path, exc)
        return False


def build_script_path(script_dir, name):
    return os.path.join(script_dir, name)
