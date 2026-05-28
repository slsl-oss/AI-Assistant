import asyncio

from utils.logger_handler import logger
from langchain_mcp_adapters.client import MultiServerMCPClient



#导入外部MCP服务的工具方法
async def run_mcp():

    #配置多个MCP服务的连接
    client = MultiServerMCPClient(
        connections={
             "weather": {
            "command": "npx",                     # 使用 npx 运行
            "args": ["@mariox/weather-mcp-server"],
            "transport": "stdio",
             }
        }
    )


    #加载MCP服务的工具方法
    tools = await client.get_tools()
    logger.info(f"成功加载工具: {[tool.name for tool in tools]}")

    return tools

tools = asyncio.run(run_mcp())



# if __name__ == '__main__':
#     asyncio.run(run_mcp())
#





