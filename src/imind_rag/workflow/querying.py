from llama_index.core import VectorStoreIndex
from llama_index.core.postprocessor.llm_rerank import LLMRerank
from llama_index.core.response_synthesizers import CompactAndRefine
from llama_index.core.workflow import Context, StartEvent, StopEvent, step

from .base import RAGWorkflow
from .event import RerankEvent, RetrieverEvent


class QueryingWorkflow(RAGWorkflow):
    def __init__(self):
        super().__init__()

    @step
    async def retrieve(self, ctx: Context, ev: StartEvent) -> RetrieverEvent | None:
        query = ev.get("query")
        if not query:
            return None

        await ctx.set("query", query)

        index = VectorStoreIndex.from_vector_store(
            self.vector_store, storage_context=self.storage_context
        )

        retriever = index.as_retriever(
            similarity_top_k=2,
            sparse_top_k=2,
            vector_store_query_mode="hybrid",
            use_async=True,
        )

        nodes = await retriever.aretrieve(query)
        print(f"Retrieved {len(nodes)} nodes.")
        return RetrieverEvent(nodes=nodes)

    @step
    async def rerank(self, ctx: Context, ev: RetrieverEvent) -> RerankEvent:
        ranker = LLMRerank(choice_batch_size=5, top_n=3, llm=self.llm)
        query = await ctx.get("query", default=None)
        new_nodes = ranker.postprocess_nodes(ev.nodes, query_str=query)
        print(f"Reranked nodes to {len(new_nodes)}")
        return RerankEvent(nodes=new_nodes)

    @step
    async def synthesize(self, ctx: Context, ev: RerankEvent) -> StopEvent:
        """Return a streaming response using reranked nodes"""
        synchronizer = CompactAndRefine(llm=self.llm, streaming=True, verbose=True)
        query = await ctx.get("query", default=True)
        response = await synchronizer.asynthesize(query, nodes=ev.nodes)

        return StopEvent(result=response)
