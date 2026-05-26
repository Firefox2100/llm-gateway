import asyncio
import contextlib
from contextlib import asynccontextmanager
from fastapi import FastAPI
from sqlalchemy.ext.asyncio import create_async_engine
from sqlmodel import SQLModel
from sqlmodel.ext.asyncio.session import AsyncSession

from llm_gateway import __version__
from llm_gateway.misc.configs import CONFIG
from llm_gateway.router import openai_router, worker_router
from llm_gateway.component import Database, Dispatcher, PendingQueue, WorkerRegistry


@asynccontextmanager
async def lifespan(application: FastAPI):
    """
    Lifespan context manager for the FastAPI application.
    :param application: The FastAPI application instance.
    """
    pending_queue = PendingQueue()
    worker_registry = WorkerRegistry()
    dispatcher = Dispatcher(
        pending_queue=pending_queue,
        worker_registry=worker_registry,
    )
    engine = create_async_engine(CONFIG.database_url)

    application.state.pending_queue = pending_queue
    application.state.worker_registry = worker_registry
    application.state.dispatcher = dispatcher
    application.state.engine = engine

    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)

    async with AsyncSession(engine) as session:
        database = Database(session=session)
        for worker in await database.list_workers():
            worker_registry.register_worker(worker)

    dispatcher_task = asyncio.create_task(dispatcher.run())

    try:
        yield
    finally:
        dispatcher.close()
        dispatcher_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await dispatcher_task
        await engine.dispose()


def create_app() -> FastAPI:
    app = FastAPI(
        title='LLM Gateway',
        version=__version__,
        description='A LLM inference gateway to route requests to nodes behind NAT.',
        contact={
            'name': 'Firefox2100',
            'url': 'https://www.firefox2100.co.uk/',
            'email': 'patrick@firefox2100.co.uk',
        },
        license_info={
            'name': 'MIT',
            'url': 'https://github.com/Firefox2100/llm-gateway/blob/main/LICENSE',
        },
        openapi_tags=[],
        lifespan=lifespan,
    )

    app.include_router(openai_router)
    app.include_router(worker_router)

    return app


app = create_app()
