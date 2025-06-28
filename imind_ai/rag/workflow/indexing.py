from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.workflow import Context, StartEvent, StopEvent, step

from .base import RAGWorkflow


class IndexingWorkflow(RAGWorkflow):
    def __init__(self):
        super().__init__()

    @step
    async def ingest(self, ctx: Context, ev: StartEvent) -> StopEvent | None:
        """Entry point to ingest a document, triggered by a StartEvent with `dirname`."""
        dirname = ev.get("dirname")
        if not dirname:
            return None

        documents = SimpleDirectoryReader(dirname).load_data()
        print(documents[0])
        index = VectorStoreIndex.from_documents(
            documents=documents,
            embed_model=self.embed_model,
            storage_context=self.storage_context,
        )

        return StopEvent(result=index)
