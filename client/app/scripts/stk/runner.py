"""Minimal runner for standalone or robot execution."""

import argparse
import platform
import sys
from distutils.version import LooseVersion

import qi


def check_commandline_args(description):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--qi-url", help="connect to a specific NAOqi instance")
    return parser.parse_args()


def is_on_robot():
    return "aldebaran" in platform.platform().lower()


def _prompt(text):
    try:
        return raw_input(text)
    except NameError:
        return input(text)


def get_debug_robot():
    try:
        import qiq.config

        qiq_robot = qiq.config.defaultHost()
        if qiq_robot:
            robot = _prompt("connect to which robot? (default is %s) " % qiq_robot)
            return robot or qiq_robot
    except Exception:
        pass
    return _prompt("connect to which robot? ")


def init(qi_url=None):
    if qi_url:
        sys.argv.extend(["--qi-url", qi_url])
    else:
        args = check_commandline_args("Run the app.")
        if args.qi_url:
            qi_url = args.qi_url
        elif not is_on_robot():
            debug_robot = get_debug_robot()
            if debug_robot:
                sys.argv.extend(["--qi-url", debug_robot])
                qi_url = debug_robot
            else:
                raise RuntimeError("No robot selected")

    sys.argv[0] = str(sys.argv[0])
    if qi_url and LooseVersion(getattr(qi, "__version__", "2.5")) < LooseVersion("2.3"):
        qiapp = qi.Application(url="tcp://%s:9559" % qi_url)
    else:
        qiapp = qi.Application()
    qiapp.start()
    return qiapp


def run_activity(activity_class, service_name=None):
    qiapp = init()
    activity = activity_class(qiapp)
    service_id = None
    qi_async = getattr(qi, "async")
    try:
        if service_name:
            service_id = qiapp.session.registerService(service_name, activity)
        if hasattr(activity, "on_start"):
            qi_async(activity.on_start)
        qiapp.run()
    finally:
        if hasattr(activity, "on_stop"):
            qi_async(activity.on_stop).wait()
        if service_id:
            qiapp.session.unregisterService(service_id)


def run_service(service_class, service_name=None):
    if not service_name:
        service_name = service_class.__name__
    run_activity(service_class, service_name)
