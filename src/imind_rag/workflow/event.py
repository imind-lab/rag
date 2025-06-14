from llama_index.core.workflow import Event
from llama_index.core.schema import NodeWithScore


class RetrieverEvent(Event):
    """Result of running retrievals"""

    nodes: list[NodeWithScore]


class RerankEvent(Event):
    """Result of running reranking on retrieved nodes"""

    nodes: list[NodeWithScore]


class SetupEvent(Event):
    pass
