"""ALMemory event helpers."""

import qi


# REMOVAL (unused code)
class EventHelper(object):

    def __init__(self, session=None):
        self.session = None
        self.almemory = None
        self.handlers = {}
        if session is not None:
            self.init(session)

    def init(self, session):
        self.session = session
        self.almemory = session.service("ALMemory")

    def connect(self, event, callback):
        if event not in self.handlers:
            self.handlers[event] = (self.almemory.subscriber(event).signal, [])
        signal, connections = self.handlers[event]
        connection_id = signal.connect(callback)
        connections.append(connection_id)
        return connection_id

    def disconnect(self, event, connection_id=None):
        if event not in self.handlers:
            return
        signal, connections = self.handlers[event]
        if connection_id is not None:
            if connection_id in connections:
                signal.disconnect(connection_id)
                connections.remove(connection_id)
            return
        for current in list(connections):
            signal.disconnect(current)
        del connections[:]

    def clear(self):
        for event in list(self.handlers.keys()):
            self.disconnect(event)
