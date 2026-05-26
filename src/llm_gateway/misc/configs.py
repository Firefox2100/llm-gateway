import os
import yaml
from pydantic import BaseModel, Field


class Config(BaseModel):
    database_url: str = Field(
        ...,
        description='The database url to use.',
    )


_config_path = os.getenv('CONFIG_PATH', 'gateway.conf')
with open(_config_path, encoding='utf-8') as f:
    _config_data = yaml.safe_load(f)


CONFIG = Config(**_config_data)
