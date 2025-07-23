import asyncio
import time
from imind_ai.rag.workflow.querying import QueryingWorkflow


async def main():
    agent = QueryingWorkflow()
    start = time.perf_counter()
    response = await agent.run(
        query="How do the suppliers price under various reliability profiles"
    )
    end = time.perf_counter()
    print(f"Query time: {(end - start):.2f}")
    print(response)


if __name__ == "__main__":
    asyncio.run(main())
