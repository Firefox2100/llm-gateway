from enum import Enum, StrEnum


class ModelCapability(Enum):
    CHAT = 'chat'                   # Text completion
    TOOL_CALLING = 'tool_calling'   # Ability to call external tools/functions
    VISION = 'vision'               # Ability to accept multi-modal image inputs
    REASONING = 'reasoning'         # Internal reasoning capability


class OpenAiMessageAudioFormat(Enum):
    MP3 = 'mp3'
    WAV = 'wav'


class OpenAiMessageContentType(StrEnum):
    FILE = 'file'
    IMAGE_URL = 'image_url'
    INPUT_AUDIO = 'input_audio'
    REFUSAL = 'refusal'
    TEXT = 'text'


class OpenAiMessageImageDetail(Enum):
    AUTO = 'auto'
    HIGH = 'high'
    LOW = 'low'


class OpenAiMessageRole(StrEnum):
    ASSISTANT = 'assistant'
    DEVELOPER = 'developer'
    FUNCTION = 'function'
    SYSTEM = 'system'
    TOOL = 'tool'
    USER = 'user'


class OpenAiMessageToolCallType(StrEnum):
    CUSTOM = 'custom'
    FUNCTION = 'function'


class OpenAiResponseAudioFormat(Enum):
    AAC = 'aac'
    FLAC = 'flac'
    MP3 = 'mp3'
    OPUS = 'opus'
    PCM16 = 'pcm16'
    WAV = 'wav'


class WebsocketMessageType(StrEnum):
    CHAT_COMPLETION_REQUEST = 'chat_completion_request'
    CHAT_COMPLETION_RESPONSE = 'chat_completion_response'
    CHAT_COMPLETION_ERROR = 'chat_completion_error'
    HEARTBEAT = 'heartbeat'


class WorkerType(Enum):
    OLLAMA = 'ollama'
