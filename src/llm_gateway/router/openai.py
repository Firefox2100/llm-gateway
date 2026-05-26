import asyncio
from fastapi import APIRouter
from fastapi.responses import JSONResponse

from llm_gateway.model.openai.chat_completion import ChatCompletionRequest
from .utils import pending_queue_dep


openai_router = APIRouter(
    prefix='/openai',
    tags=['OpenAI'],
)


@openai_router.post('/v1/chat/completions')
async def chat_completion(request: ChatCompletionRequest,
                          pending_queue: pending_queue_dep,
                          ):
    internal_request = request.to_internal()
    future = pending_queue.add(internal_request)

    try:
        response = await future
    except asyncio.CancelledError:
        pending_queue.remove(internal_request.id)
        raise

    return JSONResponse(content=response)
