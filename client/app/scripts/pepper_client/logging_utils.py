import json


def safe_json(data):
    try:
        return json.dumps(data, sort_keys=True)
    except Exception:
        return str(data)
