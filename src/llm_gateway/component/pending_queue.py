import asyncio

from llm_gateway.model.chat_completion import ChatCompletionRequest


class PendingQueue:
    """
    The queue for all pending requests.
    """
    def __init__(self):
        self._pending_requests: dict[str, ChatCompletionRequest] = {}
        self._futures: dict[str, asyncio.Future] = {}
        self._in_flight: dict[str, int] = {}
        self._order: list[str] = []

    def add(self, request: ChatCompletionRequest) -> asyncio.Future:
        self._pending_requests[request.id] = request
        self._futures[request.id] = asyncio.get_running_loop().create_future()
        self._order.append(request.id)
        return self._futures[request.id]

    def remove(self, request_id: str):
        self._pending_requests.pop(request_id, None)
        future = self._futures.pop(request_id, None)
        if future and not future.done():
            future.cancel()
        self._in_flight.pop(request_id, None)
        if request_id in self._order:
            self._order.remove(request_id)

    def mark_dispatched(self, request_id: str, worker_id: int):
        if request_id in self._order:
            self._order.remove(request_id)
        self._in_flight[request_id] = worker_id

    def requeue(self, request_id: str):
        if request_id not in self._pending_requests:
            return

        self._in_flight.pop(request_id, None)
        if request_id not in self._order:
            self._order.append(request_id)

    def requeue_worker_requests(self, worker_id: int) -> list[str]:
        request_ids = [
            request_id
            for request_id, assigned_worker_id in self._in_flight.items()
            if assigned_worker_id == worker_id
        ]
        for request_id in request_ids:
            self.requeue(request_id)

        return request_ids

    def complete(self, request_id: str, response: dict):
        future = self._futures.get(request_id)
        if future and not future.done():
            future.set_result(response)
        self._cleanup(request_id)

    def fail(self, request_id: str, exc: Exception):
        future = self._futures.get(request_id)
        if future and not future.done():
            future.set_exception(exc)
        self._cleanup(request_id)

    def _cleanup(self, request_id: str):
        self._pending_requests.pop(request_id, None)
        self._futures.pop(request_id, None)
        self._in_flight.pop(request_id, None)
        if request_id in self._order:
            self._order.remove(request_id)

    def iter_pending(self):
        for request_id in list(self._order):
            request = self._pending_requests.get(request_id)
            if request:
                yield request
