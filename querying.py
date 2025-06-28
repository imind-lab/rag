import asyncio
import time
from imind_ai.rag.workflow.querying import QueryingWorkflow


async def main():
    agent = QueryingWorkflow()
    start = time.perf_counter()
    response = await agent.run(query="课文草原在第几页？主要讲了什么？")
    end = time.perf_counter()
    print(f"Query time: {(end - start):.2f}")
    print(response)


if __name__ == "__main__":
    asyncio.run(main())
