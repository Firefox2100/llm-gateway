class LlmGatewayError(Exception):
    def __init__(self,
                 status_code: int = 500,
                 message: str = 'Internal Server Error',
                 ):
        super().__init__(message)

        self._status_code = status_code
        self._message = message


class WorkerNotFound(LlmGatewayError):
    def __init__(self,
                 worker_id: int,
                 ):
        super().__init__(
            status_code=404,
            message=f'Worker with ID {worker_id} not found.'
        )
