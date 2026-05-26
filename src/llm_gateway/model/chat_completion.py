from uuid import uuid4
from typing import Any
from pydantic import BaseModel, Field

from llm_gateway.misc.enums import ModelCapability


class ChatCompletionRequest(BaseModel):
    """
    Internal model for chat completion requests.

    This model contains more request parameters and capabilities than other compatibility API to maintain
    maximum compatibility.
    """
    id: str = Field(
        default_factory=lambda: str(uuid4()),
        description='The unique identifier for the request.',
    )
    model: str = Field(
        ...,
        description='The model to use.',
    )
    payload: dict[str, Any] = Field(
        default_factory=dict,
        description='The original API-compatible request payload to forward to the worker.',
    )

    @property
    def requested_capabilities(self) -> list[ModelCapability]:
        capabilities = {ModelCapability.CHAT}

        if self.payload.get('tools') or self.payload.get('functions'):
            capabilities.add(ModelCapability.TOOL_CALLING)

        for message in self.payload.get('messages', []):
            content = message.get('content')
            if not isinstance(content, list):
                continue

            for part in content:
                if not isinstance(part, dict):
                    continue

                if part.get('type') == 'image_url':
                    capabilities.add(ModelCapability.VISION)

        return list(capabilities)
