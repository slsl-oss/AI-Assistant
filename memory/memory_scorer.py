import re
from model.factory import chat_model
from utils.logger_handler import logger

SCORE_PROMPT = """评估这条信息对用户长期画像的重要程度（0-1）：
0.7-1.0：稳定特征（身体数据、长期偏好、职业、过敏等）
0.3-0.7：一般性偏好
0-0.3：临时性/一次性信息

信息：{memory_text}

只返回数字。"""


def score_memory(memory_text: str) -> float:
    try:
        resp = chat_model.invoke(SCORE_PROMPT.format(memory_text=memory_text))
        match = re.search(r"([01](?:\.\d+)?)", resp.content.strip())
        if match:
            return max(0.0, min(1.0, float(match.group(1))))
        return 0.5
    except Exception as e:
        logger.warning(f"[MemoryScorer] 评分失败: {e}")
        return 0.5


def score_memories(memories: list[dict]) -> list[dict]:
    """批量评分 [{memory_hash, memory_text}] → [{memory_hash, importance}]"""
    return [{"memory_hash": m["memory_hash"], "importance": score_memory(m["memory_text"])}
            for m in memories]
