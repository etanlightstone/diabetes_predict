from pydantic_ai import Agent
from pydantic_ai.mcp import MCPServerHTTP
import asyncio

server = MCPServerHTTP(url='http://localhost:8888/mcp')  
agent = Agent('openai:gpt-4.1-mini', mcp_servers=[server])  


async def main():
    async with agent.run_mcp_servers():  
        result = await agent.run('What attributes have the strongest correlation?')
    print(result.output)

if __name__ == "__main__":
    asyncio.run(main())