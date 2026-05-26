import asyncio
import contextlib

from llm_gateway.model.websocket_message import WebsocketChatCompletionRequest
from .pending_queue import PendingQueue
from .worker_registry import WorkerRegistry


class Dispatcher:
    """
    Dispatches queued requests to connected workers with matching capacity.
    """
    def __init__(self,
                 pending_queue: PendingQueue,
                 worker_registry: WorkerRegistry,
                 poll_interval: float = 0.1,
                 ):
        self._pending_queue = pending_queue
        self._worker_registry = worker_registry
        self._poll_interval = poll_interval
        self._closed = asyncio.Event()

    async def run(self):
        while not self._closed.is_set():
            dispatched = await self.dispatch_once()
            if not dispatched:
                with contextlib.suppress(asyncio.TimeoutError):
                    await asyncio.wait_for(self._closed.wait(), timeout=self._poll_interval)

    def close(self):
        self._closed.set()

    async def dispatch_once(self) -> int:
        dispatched = 0

        for request in list(self._pending_queue.iter_pending()):
            workers = self._worker_registry.available_workers_for(request)
            if not workers:
                continue

            worker = workers[0]
            session = self._worker_registry.get_session(worker.id)
            if not session:
                continue

            try:
                self._worker_registry.mark_request_started(
                    worker_id=worker.id,
                    request_id=request.id,
                )
                self._pending_queue.mark_dispatched(
                    request_id=request.id,
                    worker_id=worker.id,
                )
                message = WebsocketChatCompletionRequest(request=request)
                await session.send(message.model_dump(mode='json', exclude_none=True))
                dispatched += 1
            except Exception:
                self._worker_registry.mark_request_finished(
                    worker_id=worker.id,
                    request_id=request.id,
                )
                self._pending_queue.requeue(request.id)

        return dispatched
