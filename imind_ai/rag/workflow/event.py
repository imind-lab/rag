from llama_index.core.workflow import Event
from llama_index.core.schema import NodeWithScore


class RetrieveEvent(Event):
    """Result of running retrievals"""

    nodes: list[NodeWithScore]


class RerankEvent(Event):
    """Result of running reranking on retrieved nodes"""

    nodes: list[NodeWithScore]


class MetadataReplacementEvent(Event):
    nodes: list[NodeWithScore]


class QueryRewriteEvent(Event):
    pass
