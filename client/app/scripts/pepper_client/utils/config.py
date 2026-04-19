import copy
import json
import os

from pepper_client.interaction import speech_policy


DEFAULT_CONFIG = {
    "app": {
        "app_id": "PepperGroundedClient",
        "service_name": "PepperGroundedClient",
    },
    "server": {
        "base_url": "http://127.0.0.1:8000",
        "caption_path": "/api/v1/caption",
        "detect_path": "/api/v1/detect",
        "detect_panorama_path": "/api/v1/detect/panorama",
        "chat_path": "/api/v1/chat",
        "pregenerate_qa_path": "/api/v1/chat/pregenerate_qa",
        "config_patch_path": "/api/v1/config",
        "memory_summary_path": "/api/v1/memory/summary",
        "memory_reset_path": "/api/v1/memory/reset",
        "publish": True,
        "verify_tls": True,
        "caption_timeout_seconds": 20,
        "detect_timeout_seconds": 60,
        "memory_timeout_seconds": 20,
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
    "dialog": {
        "enable_dynamic_memory_concepts": True,
        "language": "auto",
        "memory_objects_max": 100,
        "memory_attributes_max": 100,
        "memory_relations_max": 100,
        "memory_cached_questions_max": 60,
        "refresh_after_detect": True,
        "refresh_after_scan": True,
        "refresh_after_reset": True,
    },
    "panorama": {
        "enabled": True,
        "mode": "panorama_detect",
        "stick_together": True,
        "summary_after_scan": True,
        "render_limit": 5,
    },
    "tablet": {
        "memory_render_limit": 5,
        "local_app_name": "pepper-grounded-client",
        "bridge_retry_attempts": 12,
        "bridge_retry_interval_seconds": 0.25,
        "pregenerated_questions_count": 5,
        "fake_tablet": False,
        "fake_host": "127.0.0.1",
        "fake_port": 8766,
        "fake_poll_interval_ms": 500,
    },
}

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
    config.pop("language", None)
    base_url = str(config["server"].get("base_url") or "").strip()
    if base_url.endswith("/"):
        base_url = base_url[:-1]
    config["server"]["base_url"] = base_url

    scan_yaws = config["capture"].get("scan_yaws_deg") or [-35, 0, 35]
    if not isinstance(scan_yaws, list) or not scan_yaws:
        scan_yaws = [-35, 0, 35]
    config["capture"]["scan_yaws_deg"] = scan_yaws

    dialog = config.setdefault("dialog", {})
    dialog["enable_dynamic_memory_concepts"] = bool(
        dialog.get("enable_dynamic_memory_concepts", True)
    )
    dialog.pop("language_code", None)
    dialog["language"] = speech_policy.normalize_dialog_language(
        dialog.get("language")
    )
    for key in (
        "memory_objects_max",
        "memory_attributes_max",
        "memory_relations_max",
        "memory_cached_questions_max",
    ):
        try:
            dialog[key] = max(1, int(dialog.get(key, 100)))
        except Exception:
            dialog[key] = 100
    for key in ("refresh_after_detect", "refresh_after_scan", "refresh_after_reset"):
        dialog[key] = bool(dialog.get(key, True))

    panorama = config.setdefault("panorama", {})
    panorama["enabled"] = bool(panorama.get("enabled", True))
    mode = str(panorama.get("mode") or "panorama_detect").strip().lower()
    if mode not in ("panorama_detect", "sequential_detect"):
        mode = "panorama_detect"
    panorama["mode"] = mode
    panorama["stick_together"] = bool(panorama.get("stick_together", True))
    panorama["summary_after_scan"] = bool(panorama.get("summary_after_scan", True))
    try:
        panorama["render_limit"] = max(1, int(panorama.get("render_limit", 5)))
    except Exception:
        panorama["render_limit"] = 5

    tablet = config.setdefault("tablet", {})
    tablet["local_app_name"] = str(
        tablet.get("local_app_name") or "pepper-grounded-client"
    ).strip()
    try:
        tablet["memory_render_limit"] = max(1, int(tablet.get("memory_render_limit", 5)))
    except Exception:
        tablet["memory_render_limit"] = 5
    try:
        tablet["bridge_retry_attempts"] = max(
            1, int(tablet.get("bridge_retry_attempts", 12))
        )
    except Exception:
        tablet["bridge_retry_attempts"] = 12
    try:
        tablet["bridge_retry_interval_seconds"] = max(
            0.0, float(tablet.get("bridge_retry_interval_seconds", 0.25))
        )
    except Exception:
        tablet["bridge_retry_interval_seconds"] = 0.25
    try:
        tablet["pregenerated_questions_count"] = max(
            1, int(tablet.get("pregenerated_questions_count", 5))
        )
    except Exception:
        tablet["pregenerated_questions_count"] = 5
    tablet["fake_tablet"] = bool(tablet.get("fake_tablet", False))
    tablet["fake_host"] = str(tablet.get("fake_host") or "127.0.0.1").strip()
    if not tablet["fake_host"]:
        tablet["fake_host"] = "127.0.0.1"
    try:
        tablet["fake_port"] = int(tablet.get("fake_port", 8766))
    except Exception:
        tablet["fake_port"] = 8766
    if tablet["fake_port"] < 1 or tablet["fake_port"] > 65535:
        tablet["fake_port"] = 8766
    try:
        tablet["fake_poll_interval_ms"] = int(tablet.get("fake_poll_interval_ms", 500))
    except Exception:
        tablet["fake_poll_interval_ms"] = 500
    if tablet["fake_poll_interval_ms"] < 100:
        tablet["fake_poll_interval_ms"] = 100

    return config


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
