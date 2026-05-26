from typing import Annotated
from fastapi import Request, Depends
from sqlmodel.ext.asyncio.session import AsyncSession

from llm_gateway.component import Database, PendingQueue, WorkerRegistry


async def get_session(request: Request):
    engine = request.app.state.engine
    async with AsyncSession(engine) as session:
        yield session


session_dep = Annotated[AsyncSession, Depends(get_session)]


async def get_database(session: session_dep,
                       ) -> Database:
    return Database(session=session)


def get_pending_queue(request: Request) -> PendingQueue:
    pending_queue = request.app.state.pending_queue

    return pending_queue


def get_worker_registry(request: Request) -> WorkerRegistry:
    worker_registry = request.app.state.worker_registry

    return worker_registry


pending_queue_dep = Annotated[PendingQueue, Depends(get_pending_queue)]
worker_registry_dep = Annotated[WorkerRegistry, Depends(get_worker_registry)]
database_dep = Annotated[Database, Depends(get_database)]
