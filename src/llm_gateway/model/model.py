from pydantic import BaseModel, Field

from llm_gateway.misc.enums import ModelCapability


class Model(BaseModel):
    """
    An registered inference model in the system. It may not match the exact model name used on the
    worker backend, but a convenient identifier for incoming requests to use.
    """
    name: str = Field(
        ...,
        description='The name of the model.',
    )
    capabilities: list[ModelCapability] = Field(
        ...,
        description='The capabilities of the model.',
        min_length=1,
    )
    streaming: bool = Field(
        False,
        description='Whether the model supports streaming output.',
    )
    max_tokens: int = Field(
        ...,
        description='The maximum number of tokens that the model can generate in a single response.',
    )
