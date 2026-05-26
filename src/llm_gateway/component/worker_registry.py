import asyncio
from datetime import datetime, timezone
from argon2 import PasswordHasher
from argon2.exceptions import VerifyMismatchError
from fastapi import WebSocket

from llm_gateway.model.worker import Worker, WorkerState
from llm_gateway.model.chat_completion import ChatCompletionRequest


class WorkerSession:
    def __init__(self, worker_id: int, websocket: WebSocket):
        self.worker_id = worker_id
        self.websocket = websocket
        self.send_queue: asyncio.Queue[dict] = asyncio.Queue(maxsize=100)
        self.closed = asyncio.Event()

    async def send(self, message: dict):
        await self.send_queue.put(message)

    async def sender_loop(self):
        while not self.closed.is_set():
            message = await self.send_queue.get()
            await self.websocket.send_json(message)

    async def receiver_loop(self):
        while not self.closed.is_set():
            await self.websocket.receive_json()


class WorkerRegistry:
    """
    A registry for all workers in the system.
    """
    def __init__(self):
        self._password_hasher = PasswordHasher()

        # workers are for all known workers, including the ones that are not connected/down
        self._workers: dict[int, Worker] = {}

        # worker_states are for all connected workers and will be removed when the worker is disconnected/down
        self._worker_states: dict[int, WorkerState] = {}

        # sessions are for all connected workers
        self._sessions: dict[int, WorkerSession] = {}

    @property
    def workers(self) -> dict[int, Worker]:
        return self._workers.copy()

    @property
    def worker_states(self) -> dict[int, WorkerState]:
        return self._worker_states.copy()

    def register_worker(self, worker: Worker):
        self._workers[worker.id] = worker

    def update_worker_state(self,
                            worker_state: WorkerState,
                            ):
        worker_id = worker_state.worker.id
        if worker_id not in self._workers:
            raise ValueError(f'Worker with ID {worker_id} not found.')

        self._worker_states[worker_id] = worker_state

    def register_session(self, session: WorkerSession):
        worker_id = session.worker_id
        if worker_id not in self._workers:
            raise ValueError(f'Worker with ID {worker_id} not found.')

        self._sessions[worker_id] = session

    def get_session(self, worker_id: int) -> WorkerSession | None:
        return self._sessions.get(worker_id)

    def remove_worker_state(self, worker_id: int):
        self._worker_states.pop(worker_id, None)

    def unregister_session(self, session: WorkerSession):
        worker_id = session.worker_id
        self._sessions.pop(worker_id, None)

    def mark_request_started(self,
                             worker_id: int,
                             request_id: str,
                             ):
        state = self._worker_states.get(worker_id)
        if not state:
            raise ValueError(f'Worker with ID {worker_id} has no active state.')

        state.active_requests.add(request_id)

    def mark_request_finished(self,
                              worker_id: int,
                              request_id: str,
                              ):
        state = self._worker_states.get(worker_id)
        if not state:
            return

        state.active_requests.discard(request_id)

    def mark_worker_reported(self,
                             worker_id: int,
                             ):
        state = self._worker_states.get(worker_id)
        if not state:
            return

        state.last_report = datetime.now(timezone.utc)
        state.healthy = True

    def authenticate_worker(self,
                            worker_id: int,
                            worker_secret: str,
                            ):
        """
        Authenticate a worker by verifying the worker secret with the stored secret hash.
        :param worker_id: The id of the worker.
        :param worker_secret: The secret to authenticate the worker with.
        :return: True if the worker is authenticated, False otherwise.
        """
        worker = self._workers.get(worker_id)
        if not worker:
            return False

        try:
            self._password_hasher.verify(
                hash=worker.secret_hash,
                password=worker_secret,
            )
            return True
        except VerifyMismatchError:
            return False

    def available_workers_for(self,
                              request: ChatCompletionRequest,
                              ) -> list[Worker]:
        available_workers = []

        for state in self._worker_states.values():
            if not state.healthy:
                continue

            if len(state.active_requests) >= state.max_concurrency:
                continue

            possible_model = next((model for model in state.models if model.name == request.model), None)
            if not possible_model:
                continue

            capabilities_match = True
            for c in request.requested_capabilities:
                if c not in possible_model.capabilities:
                    capabilities_match = False

            if not capabilities_match:
                continue

            available_workers.append(state.worker)

        return available_workers
