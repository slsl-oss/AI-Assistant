
from langchain_mcp_adapters.client import MultiServerMCPClient
import asyncio
from utils.logger_handler import logger



#导入外部MCP服务的工具方法
async def run_mcp():

    #配置多个MCP服务的连接
    client = MultiServerMCPClient(
        connections={
            "weather": {  # 连接名称   获取天气服务
                "command": "npx",  # 调用npm包管理器
                "args": ["@mariox/weather-mcp-server"],  # 要运行的包名
                "transport": "stdio",  # 传输协议，stdio表示标准输入输出
            }
        }
    )


    #加载MCP服务的工具方法
    tools = await client.get_tools()
    return tools

weather_tools = asyncio.run(run_mcp())
logger.info(f"成功加载天气agent的工具: {[tool.name for tool in weather_tools]}")


