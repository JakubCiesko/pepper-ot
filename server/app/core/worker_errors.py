class WorkerError(RuntimeError):
    pass


class WorkerUnavailableError(WorkerError):
    pass


class WorkerStartupTimeoutError(WorkerError):
    pass


class WorkerQueueFullError(WorkerError):
    pass


class WorkerCircuitOpenError(WorkerError):
    pass


class WorkerProtocolError(WorkerError):
    pass
