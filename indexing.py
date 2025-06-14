import asyncio
import time
from imind_rag.workflow.indexing import IndexingWorkflow


async def main():
    agent = IndexingWorkflow()
    start = time.perf_counter()
    index = await agent.run(dirname="data")
    end = time.perf_counter()
    print(f"Index time: {(end - start):.2f}")
    print(index)


if __name__ == "__main__":
    asyncio.run(main())
