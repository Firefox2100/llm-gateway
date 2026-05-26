from sqlmodel import Field, SQLModel
from sqlmodel.ext.asyncio.session import AsyncSession
from sqlmodel import select

from llm_gateway.misc.enums import WorkerType
from llm_gateway.misc.errors import WorkerNotFound
from llm_gateway.model.worker import Worker


class WorkerStorage(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    type: str = Field(index=True)
    secret_hash: str


class Database:
    def __init__(self,
                 session: AsyncSession,
                 ):
        self._session = session

    async def get_worker(self,
                         worker_id: int,
                         ) -> Worker:
        worker = await self._session.get(WorkerStorage, worker_id)
        if not worker:
            raise WorkerNotFound(worker_id=worker_id)

        if worker.id is None:
            raise ValueError('Worker ID is None')

        return Worker(
            id=worker.id,
            type=WorkerType(worker.type),
            secret_hash=worker.secret_hash,
        )

    async def list_workers(self) -> list[Worker]:
        result = await self._session.exec(select(WorkerStorage))
        workers = result.all()

        return [
            Worker(
                id=worker.id,
                type=WorkerType(worker.type),
                secret_hash=worker.secret_hash,
            )
            for worker in workers
            if worker.id is not None
        ]
