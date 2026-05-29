from langchain_core.tools import tool
from langchain_core.runnables.config import var_child_runnable_config
from utils.config_handler import memory_conf
from utils.logger_handler import logger
from memory.mem0_service import mem0_service


def _get_current_user_id():
    """从 LangGraph 运行时上下文获取当前 user_id"""
    try:
        config = var_child_runnable_config.get()
        return config.get("configurable", {}).get("user_id", memory_conf["default_user_id"])
    except Exception:
        return memory_conf["default_user_id"]


@tool(description="将重要信息保存到长期记忆中，跨会话记住用户偏好和关键事实。当用户透露个人偏好（风格/尺码/颜色/品牌/面料/预算）、身体数据、职业等信息时调用。")
def save_memory(content: str) -> str:
    user_id = _get_current_user_id()
    mem0_service.add(content, user_id=user_id)
    logger.info(f"[save_memory] 已保存记忆，user_id: {user_id}")
    return "记忆已保存"


@tool(description="搜索长期记忆，查找用户的偏好和历史信息。在回答需要个性化推荐的问题前，先调用此方法了解用户。")
def search_memory(query: str) -> str:
    user_id = _get_current_user_id()
    results = mem0_service.search(query, user_id=user_id)
    formatted = mem0_service.format_memories_for_prompt(results)
    logger.info(f"[search_memory] 搜索记忆完成，结果数: {len(results)}")
    return formatted or "未找到相关记忆"


memory_tools = [save_memory, search_memory]
logger.info(f"成功加载记忆工具: {[t.name for t in memory_tools]}")
