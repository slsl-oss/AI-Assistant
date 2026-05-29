import os
from mem0 import Memory
from utils.config_handler import memory_conf
from utils.path_tool import get_abs_path
from utils.logger_handler import logger


class Mem0Service:
    """Mem0 长期记忆服务单例"""

    _instance = None

    def __init__(self):
        chroma_path = get_abs_path(memory_conf["chroma_path"])
        history_db_path = get_abs_path(memory_conf["history_db_path"])

        with open(get_abs_path("prompts/memory_extractor_prompt.txt"), "r", encoding="utf-8") as f:
            custom_instructions = f.read()

        config = {
            "llm": {
                "provider": "openai",
                "config": {
                    "model": "qwen-max",
                    "api_key": os.getenv("DASHSCOPE_API_KEY"),
                    "openai_base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1"
                }
            },
            "embedder": {
                "provider": "openai",
                "config": {
                    "model": "text-embedding-v4",
                    "api_key": os.getenv("DASHSCOPE_API_KEY"),
                    "openai_base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1"
                }
            },
            "vector_store": {
                "provider": "chroma",
                "config": {
                    "collection_name": memory_conf["collection_name"],
                    "path": chroma_path
                }
            },
            "history_db_path": history_db_path,
            "custom_instructions": custom_instructions
        }

        self.memory = Memory.from_config(config)
        logger.info(f"[Mem0Service] 初始化完成，collection: {memory_conf['collection_name']}")

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def add(self, messages, user_id=None, metadata=None):
        """
        添加记忆，Mem0 自动从消息中提取关键事实

        Args:
            messages: 消息列表 [{"role": "user", "content": "..."}, ...] 或单条消息
            user_id: 用户ID
            metadata: 额外元数据
        """
        user_id = user_id or memory_conf["default_user_id"]
        try:
            result = self.memory.add(messages, user_id=user_id, metadata=metadata)
            logger.info(f"[Mem0Service] 添加记忆成功，user_id: {user_id}")
            return result
        except Exception as e:
            logger.error(f"[Mem0Service] 添加记忆失败: {e}")
            return None

    def search(self, query, user_id=None, k=None):
        """
        语义搜索历史记忆

        Args:
            query: 搜索查询
            user_id: 用户ID
            k: 返回结果数量

        Returns:
            搜索结果列表，每项为 {"memory": "...", ...}
        """
        user_id = user_id or memory_conf["default_user_id"]
        k = k or memory_conf["k"]
        try:
            result = self.memory.search(query, filters={"user_id": user_id}, limit=k)
            # Mem0 2.0 返回 {"results": [...]}
            if isinstance(result, dict) and "results" in result:
                items = result["results"]
            elif isinstance(result, list):
                items = result
            else:
                items = []
            logger.info(f"[Mem0Service] 搜索记忆成功，query: {query[:50]}..., 结果数: {len(items)}")
            return items
        except Exception as e:
            logger.error(f"[Mem0Service] 搜索记忆失败: {e}")
            return []

    def get_all(self, user_id=None):
        """获取用户全部记忆"""
        user_id = user_id or memory_conf["default_user_id"]
        try:
            return self.memory.get_all(filters={"user_id": user_id})
        except Exception as e:
            logger.error(f"[Mem0Service] 获取全部记忆失败: {e}")
            return []

    def format_memories_for_prompt(self, search_results):
        """将搜索结果格式化为 Prompt 上下文"""
        if not search_results:
            return ""

        lines = ["[长期记忆 - 用户偏好和历史信息]"]
        for item in search_results:
            if isinstance(item, dict):
                memory_text = item.get("memory", "")
            elif isinstance(item, str):
                memory_text = item
            else:
                memory_text = str(item)
            
            if memory_text:
                lines.append(f"- {memory_text}")

        return "\n".join(lines)


mem0_service = Mem0Service.get_instance()
