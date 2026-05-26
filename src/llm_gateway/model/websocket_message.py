from typing import Any, Literal, Annotated, Union
from pydantic import BaseModel, Field

from llm_gateway.misc.enums import WebsocketMessageType
from .chat_completion import ChatCompletionRequest


class WebsocketChatCompletionRequest(BaseModel):
    type: Literal[WebsocketMessageType.CHAT_COMPLETION_REQUEST] = Field(
        WebsocketMessageType.CHAT_COMPLETION_REQUEST,
        description='The type of the websocket message, in this case chat_completion_request',
    )
    request: ChatCompletionRequest = Field(
        ...,
        description='The chat completion request payload',
    )


class WebsocketChatCompletionResponse(BaseModel):
    type: Literal[WebsocketMessageType.CHAT_COMPLETION_RESPONSE] = Field(
        WebsocketMessageType.CHAT_COMPLETION_RESPONSE,
        description='The type of the websocket message, in this case chat_completion_response',
    )
    request_id: str = Field(
        ...,
        description='The ID of the request this response completes.',
    )
    response: dict[str, Any] = Field(
        ...,
        description='The API-compatible response payload.',
    )


class WebsocketChatCompletionError(BaseModel):
    type: Literal[WebsocketMessageType.CHAT_COMPLETION_ERROR] = Field(
        WebsocketMessageType.CHAT_COMPLETION_ERROR,
        description='The type of the websocket message, in this case chat_completion_error',
    )
    request_id: str = Field(
        ...,
        description='The ID of the request this error completes.',
    )
    message: str = Field(
        ...,
        description='The error message returned by the worker.',
    )


class WebsocketHeartbeat(BaseModel):
    type: Literal[WebsocketMessageType.HEARTBEAT] = Field(
        WebsocketMessageType.HEARTBEAT,
        description='The type of the websocket message, in this case heartbeat',
    )


WebsocketMessage = Annotated[
    Union[
        WebsocketChatCompletionRequest,
        WebsocketChatCompletionResponse,
        WebsocketChatCompletionError,
        WebsocketHeartbeat,
    ],
    Field(discriminator='type')
]
