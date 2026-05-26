import asyncio
import json
import os
import time
from typing import Any
from urllib import request as urllib_request
import yaml
from pydantic import BaseModel, Field

from llm_gateway.model.chat_completion import ChatCompletionRequest
from llm_gateway.model.model import Model
from .worker import Worker


OLLAMA_CONFIG_PATH = os.getenv('OLLAMA_CONFIG_PATH', 'ollama.conf')


class OllamaConfig(BaseModel):
    worker_id: int = Field(
        ...,
        description='The worker ID registered with the gateway.',
    )
    url: str = Field(
        ...,
        description='The URL of the Ollama server, e.g. http://localhost:11434',
    )
    models: list[Model] = Field(
        ...,
        description='The models that are available on the Ollama server. Note that this does not have to match '
                    'the exact models that are available on the Ollama server, but an exposed capability '
                    'wrapper. For example, some models may be omitted, some may have lower context window, etc.',
    )
    model_name_mapping: dict[str, str] = Field(
        ...,
        description='A mapping from the model name exposed by the gateway to the model name on the Ollama server.',
    )
    max_concurrency: int = Field(
        1,
        description='The maximum number of concurrent requests that the worker can handle.',
    )

    gateway_url: str = Field(
        ...,
        description='The URL of the gateway server, e.g. http://localhost:5769',
    )


class OllamaWorker(Worker):
    def __init__(self,
                 worker_id: int,
                 gateway_url: str,
                 gateway_secret: str,
                 models: list[Model],
                 model_name_mapping: dict[str, str],
                 max_concurrency: int,
                 ollama_url: str,
                 ):
        super().__init__(
            worker_id=worker_id,
            gateway_url=gateway_url,
            gateway_secret=gateway_secret,
            models=models,
            max_concurrency=max_concurrency,
        )
        self._model_name_mapping = model_name_mapping
        self._ollama_url = ollama_url.rstrip('/')

    async def process_chat_completion(self,
                                      request: ChatCompletionRequest,
                                      ) -> dict:
        return await asyncio.to_thread(self._process_chat_completion_sync, request)

    def _process_chat_completion_sync(self,
                                      request: ChatCompletionRequest,
                                      ) -> dict:
        payload = request.payload.copy()
        model_name = self._model_name_mapping.get(request.model, request.model)
        ollama_payload = {
            'model': model_name,
            'messages': payload.get('messages', []),
            'stream': False,
        }

        if 'temperature' in payload:
            ollama_payload['options'] = {
                'temperature': payload['temperature'],
            }

        data = json.dumps(ollama_payload).encode('utf-8')
        http_request = urllib_request.Request(
            url=f'{self._ollama_url}/api/chat',
            data=data,
            headers={'Content-Type': 'application/json'},
            method='POST',
        )

        with urllib_request.urlopen(http_request, timeout=300) as response:
            ollama_response: dict[str, Any] = json.loads(response.read().decode('utf-8'))

        message = ollama_response.get('message', {})
        content = message.get('content', '')

        return {
            'id': request.id,
            'object': 'chat.completion',
            'created': int(time.time()),
            'model': request.model,
            'choices': [
                {
                    'index': 0,
                    'message': {
                        'role': 'assistant',
                        'content': content,
                    },
                    'finish_reason': 'stop',
                }
            ],
            'usage': {
                'prompt_tokens': ollama_response.get('prompt_eval_count', 0),
                'completion_tokens': ollama_response.get('eval_count', 0),
                'total_tokens': (
                    ollama_response.get('prompt_eval_count', 0)
                    + ollama_response.get('eval_count', 0)
                ),
            },
        }


if __name__ == '__main__':
    with open(OLLAMA_CONFIG_PATH, encoding='utf-8') as f:
        config_data = yaml.safe_load(f)

    config = OllamaConfig(**config_data)
    gateway_secret = os.getenv('LLM_GATEWAY_SECRET')
    if not gateway_secret:
        raise ValueError('LLM_GATEWAY_SECRET is not set')

    worker = OllamaWorker(
        worker_id=config.worker_id,
        gateway_url=config.gateway_url,
        gateway_secret=gateway_secret,
        models=config.models,
        model_name_mapping=config.model_name_mapping,
        max_concurrency=config.max_concurrency,
        ollama_url=config.url,
    )
    asyncio.run(worker.run())
