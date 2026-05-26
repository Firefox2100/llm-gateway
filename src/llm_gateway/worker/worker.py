import asyncio
import json
from abc import ABC, abstractmethod
from urllib.parse import urlparse
import websockets

from llm_gateway.misc.enums import WebsocketMessageType
from llm_gateway.model.chat_completion import ChatCompletionRequest
from llm_gateway.model.worker import WorkerCapability
from llm_gateway.model.model import Model


class Worker(ABC):
    """
    Base class for all workers, this class contains the common methods and attributes.
    """
    def __init__(self,
                 worker_id: int,
                 gateway_url: str,
                 gateway_secret: str,
                 models: list[Model],
                 max_concurrency: int,
                 ):
        self._worker_id = worker_id
        self._gateway_url = gateway_url
        self._gateway_secret = gateway_secret

        self._models = models
        self._max_concurrency = max_concurrency

    async def run(self):
        # Convert the http/https URL to a websocket URL
        parsed_url = urlparse(self._gateway_url)
        scheme = 'wss' if parsed_url.scheme == 'https' else 'ws'
        websocket_url = f'{scheme}://{parsed_url.netloc}/worker/connect/{self._worker_id}'

        async with websockets.connect(
            websocket_url,
            additional_headers={
                'Authorization': f'Bearer {self._gateway_secret}',
            }
        ) as websocket:
            capability = WorkerCapability(
                models=self._models,
                max_concurrency=self._max_concurrency,
            )
            await websocket.send(capability.model_dump_json(exclude_none=True))

            semaphore = asyncio.Semaphore(self._max_concurrency)
            send_lock = asyncio.Lock()
            active_tasks: set[asyncio.Task] = set()

            async for raw_message in websocket:
                message = json.loads(raw_message)
                if message.get('type') != WebsocketMessageType.CHAT_COMPLETION_REQUEST:
                    continue

                task = asyncio.create_task(
                    self._handle_chat_completion_message(
                        websocket=websocket,
                        semaphore=semaphore,
                        send_lock=send_lock,
                        message=message,
                    )
                )
                active_tasks.add(task)
                task.add_done_callback(active_tasks.discard)

            if active_tasks:
                await asyncio.gather(*active_tasks, return_exceptions=True)

    async def _handle_chat_completion_message(self,
                                              websocket,
                                              semaphore: asyncio.Semaphore,
                                              send_lock: asyncio.Lock,
                                              message: dict,
                                              ):
        request = ChatCompletionRequest(**message['request'])

        async with semaphore:
            try:
                response = await self.process_chat_completion(request)
                async with send_lock:
                    await websocket.send(json.dumps({
                        'type': WebsocketMessageType.CHAT_COMPLETION_RESPONSE,
                        'request_id': request.id,
                        'response': response,
                    }))
            except Exception as exc:
                async with send_lock:
                    await websocket.send(json.dumps({
                        'type': WebsocketMessageType.CHAT_COMPLETION_ERROR,
                        'request_id': request.id,
                        'message': str(exc),
                    }))

    @abstractmethod
    async def process_chat_completion(self,
                                      request: ChatCompletionRequest,
                                      ) -> dict:
        """
        Process a chat completion request and return an API-compatible response payload.
        """
