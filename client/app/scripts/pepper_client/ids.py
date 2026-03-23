import uuid


def new_turn_id(prefix="turn"):
    return "%s_%s" % (prefix, uuid.uuid4().hex)


def new_frame_id(prefix="frame"):
    return "%s_%s" % (prefix, uuid.uuid4().hex)


def new_scan_id(prefix="scan"):
    return "%s_%s" % (prefix, uuid.uuid4().hex)
