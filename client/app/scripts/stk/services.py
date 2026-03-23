"""Service cache helpers."""


class ServiceCache(object):
    def __init__(self, session=None):
        self.session = None
        self.services = {}
        if session is not None:
            self.init(session)

    def init(self, session):
        self.session = session

    def __getattr__(self, service_name):
        if service_name.startswith("__"):
            raise AttributeError
        if (service_name not in self.services) or (service_name == "ALTabletService"):
            try:
                self.services[service_name] = self.session.service(service_name)
            except Exception:
                self.services[service_name] = None
        return self.services[service_name]
