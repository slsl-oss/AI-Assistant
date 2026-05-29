import asyncio

from langchain_core.tools import tool
from utils.logger_handler import logger
from rag.rag_service import RagSummarizeService
import requests
import os
from memory.memory_tools import save_memory, search_memory



# 调用api的工具
@tool(description="使用百度搜索引擎进行联网搜索")
def baidu_web_search(query: str) -> str:
    """
    使用百度搜索引擎进行联网搜索。
    当需要查询实时信息、最新新闻、天气、股票等需要联网的问题时使用。
    """
    API_KEY = os.getenv("BAIDU_API_KEY")  #系统环境变量中获取API_KEY
    # 可选：添加检查，避免出错
    if not API_KEY:
        raise ValueError("请先设置 BAIDU_API_KEY 环境变量")

    url = "https://qianfan.baidubce.com/v2/ai_search/web_search"
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "messages": [{"content": query}]
    }

    response = requests.post(url, headers=headers, json=data)

    if response.status_code == 200:
        result = response.json()
        # 根据返回格式提取搜索结果
        return str(result)
    else:
        return f"搜索失败: {response.status_code}, {response.text}"


@tool(description="从向量存储中检索参考资料")
def  rag_summarize(query: str) -> str:
    rag_service = RagSummarizeService()
    return rag_service.rag_summarize(query)




@tool(description="无入参，无返回值，调用后触发中间件，自动为特定场景填充特定的上下文信息，")
def fill_context_for_other_prompt():
    logger.info(f"[fill_context_for_other_prompt]已调用")
    return







base_tools =  [rag_summarize,fill_context_for_other_prompt,baidu_web_search,
               save_memory, search_memory]
logger.info(f"成功加载工具: {[tool.name for tool in base_tools]}")


# if __name__ == '__main__':





