from typing import Optional
from pydantic import BaseModel, ConfigDict


class KWArgs(BaseModel):
    model_config = ConfigDict(extra="allow")

    temperature: Optional[float] = None


class Model(BaseModel):
    model_config = ConfigDict(extra="ignore")

    provider: str
    model_name: str
    base_url: Optional[str] = None
    api_key: Optional[str] = None
    kwargs: Optional[KWArgs] = None


class Embedding(BaseModel):
    model_config = ConfigDict(extra="ignore")

    chunk_size: int
    chunk_overlap: int = 0
    dense: Model
    sparse: Optional[Model] = None


class VectorStore(BaseModel):
    model_config = ConfigDict(extra="ignore")

    provider: str
    collection_name: str
    host: str = "localhost"
    port: int = 6333


class Config(BaseModel):
    model_config = ConfigDict(extra="ignore")

    llm: Model
    embedding: Embedding
    vector_store: VectorStore
