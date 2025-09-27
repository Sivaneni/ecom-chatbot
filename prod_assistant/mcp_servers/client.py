import asyncio
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools

async def main():
    client = MultiServerMCPClient({
        "hybrid_search": {   # server name
            "command": "python",
            "args": [
                r"C:\Users\SIVANENI PRASANNA\ecom-prod-assistant\prod_assistant\mcp_servers\product_search_server.py"
            ],  # absolute path
            "transport": "stdio"
            # "url": "http://localhost:8000/mcp"  # URL for streamable-http
        }
    })

    # Discover tools
    async with client.session("hybrid_search") as session:
        tools = await load_mcp_tools(session)
        print("Available tools:", [t.name for t in tools])

        # Pick tools by name
        retriever_tool = next(t for t in tools if t.name == "get_product_info")
        web_tool = next(t for t in tools if t.name == "web_search")

        # --- Step 1: Try retriever first ---
        #query = "Samsung Galaxy S25 price"
        # query = "iPhone 15"
        query = "What is the issue between India vs Pakistan in Asia Cup?"
        retriever_result = await retriever_tool.ainvoke({"query": query})
        print("\nRetriever Result:\n", retriever_result)

        # --- Step 2: Fallback to web search if retriever fails ---
        if not retriever_result.strip() or "No local results found." in retriever_result:
            print("\n No local results, falling back to web search...\n")
            web_result = await web_tool.ainvoke({"query": query})
            print("Web Search Result:\n", web_result)

if __name__ == "__main__":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())