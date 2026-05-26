import asyncio
from datetime import datetime, timezone
from fastapi import APIRouter, HTTPException, WebSocket
from fastapi.websockets import WebSocketDisconnect
from pydantic import ValidationError

from llm_gateway.misc.enums import WebsocketMessageType
from llm_gateway.model.worker import WorkerCapability, WorkerState
from llm_gateway.component.worker_registry import WorkerSession
from .utils import pending_queue_dep, worker_registry_dep


worker_router = APIRouter(
    prefix='/worker',
    tags=['Worker'],
)


@worker_router.websocket('/connect/{worker_id}')
async def worker_websocket(worker_id: int,
                           websocket: WebSocket,
                           worker_registry: worker_registry_dep,
                           pending_queue: pending_queue_dep,
                           ):
    auth_header = websocket.headers.get('Authorization')

    if not auth_header or not auth_header.startswith('Bearer '):
        await websocket.close(
            code=4001,
            reason='Invalid authorization header',
        )
        return

    worker_secret = auth_header.split(' ')[1]
    if not worker_registry.authenticate_worker(
        worker_id=worker_id,
        worker_secret=worker_secret,
    ):
        await websocket.close(
            code=4001,
            reason='Invalid worker secret',
        )
        return

    await websocket.accept()
    capability_data = await websocket.receive_json()

    try:
        worker_capability = WorkerCapability(**capability_data)
    except ValidationError as e:
        await websocket.close(
            code=4002,
            reason=f'Invalid capability data: {e}',
        )
        return

    session = WorkerSession(
        worker_id=worker_id,
        websocket=websocket,
    )

    worker_registry.update_worker_state(
        worker_state=WorkerState(
            worker=worker_registry.workers[worker_id],
            models=worker_capability.models,
            max_concurrency=worker_capability.max_concurrency,
            active_requests=set(),
            healthy=True,
            last_report=datetime.now(timezone.utc),
        )
    )
    worker_registry.register_session(session)

    sender_task = asyncio.create_task(session.sender_loop())
    receiver_task = asyncio.create_task(
        _worker_receiver_loop(
            worker_id=worker_id,
            websocket=websocket,
            worker_registry=worker_registry,
            pending_queue=pending_queue,
        )
    )

    try:
        done, pending = await asyncio.wait(
            {sender_task, receiver_task},
            return_when=asyncio.FIRST_EXCEPTION,
        )

        for task in done:
            exc = task.exception()
            if exc:
                raise exc
    except WebSocketDisconnect:
        # Worker disconnected itself
        pass
    finally:
        session.closed.set()

        for task in (sender_task, receiver_task):
            task.cancel()

        worker_registry.remove_worker_state(worker_id)
        worker_registry.unregister_session(session)
        requeued_request_ids = pending_queue.requeue_worker_requests(worker_id)
        for request_id in requeued_request_ids:
            worker_registry.mark_request_finished(
                worker_id=worker_id,
                request_id=request_id,
            )


async def _worker_receiver_loop(worker_id: int,
                                websocket: WebSocket,
                                worker_registry,
                                pending_queue,
                                ):
    while True:
        message = await websocket.receive_json()
        worker_registry.mark_worker_reported(worker_id)

        message_type = message.get('type')
        if message_type == WebsocketMessageType.HEARTBEAT:
            continue

        request_id = message.get('request_id')
        if not request_id:
            continue

        if message_type == WebsocketMessageType.CHAT_COMPLETION_RESPONSE:
            pending_queue.complete(
                request_id=request_id,
                response=message.get('response', {}),
            )
            worker_registry.mark_request_finished(
                worker_id=worker_id,
                request_id=request_id,
            )
        elif message_type == WebsocketMessageType.CHAT_COMPLETION_ERROR:
            pending_queue.fail(
                request_id=request_id,
                exc=HTTPException(
                    status_code=502,
                    detail=message.get('message', 'Worker failed to process request'),
                ),
            )
            worker_registry.mark_request_finished(
                worker_id=worker_id,
                request_id=request_id,
            )
