from llama_index.core import (
    StorageContext,
    Settings,
)

from llama_index.core.workflow import Workflow
from llama_index.embeddings.fastembed import FastEmbedEmbedding
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core.node_parser import SentenceWindowNodeParser
from llama_index.llms.openai_like import OpenAILike

from qdrant_client import QdrantClient, AsyncQdrantClient

from imind_rag.config.settings import get_settings


class RAGWorkflow(Workflow):
    def __init__(self):
        super().__init__()

        settings = get_settings()
        print("settings", settings)

        client = QdrantClient(
            host=settings.vector_store.host, port=settings.vector_store.port
        )
        aclient = AsyncQdrantClient(
            host=settings.vector_store.host, port=settings.vector_store.port
        )

        self.vector_store = QdrantVectorStore(
            settings.vector_store.collection_name,
            client=client,
            aclient=aclient,
            enable_hybrid=True,
            fastembed_sparse_model=settings.embedding.sparse.model_name,
            batch_size=32,
            prefer_grpc=True,
        )

        self.storage_context = StorageContext.from_defaults(
            vector_store=self.vector_store
        )

        self.embed_model = FastEmbedEmbedding(
            model_name=settings.embedding.dense.model_name
        )

        self.llm = OpenAILike(
            model=settings.llm.model_name,
            api_base=settings.llm.base_url,
            is_chat_model=True,
            is_function_calling_model=False,
            temperature=settings.llm.kwargs.temperature,
            api_key=settings.llm.api_key,
        )

        Settings.chunk_size = settings.embedding.chunk_size
        Settings.chunk_overlap = settings.embedding.chunk_overlap
        Settings.llm = self.llm
        Settings.embed_model = self.embed_model
        Settings.node_parser = SentenceWindowNodeParser.from_defaults(
            window_size=3,
            window_metadata_key="window",
            original_text_metadata_key="original_text",
        )
