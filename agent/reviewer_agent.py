"""
审查 Agent：审查草稿质量，返回 accept/revise + 修改意见
"""
import json
from model.factory import research_agent_model
from utils.logger_handler import logger


class ReviewerAgent:
    """审查草稿质量，决定 accept 或 revise"""

    def __init__(self):
        self.model = research_agent_model

    async def review(self, section: str, draft: str) -> dict:
        """审查草稿质量。Returns: {verdict: "accept"|"revise", notes: str}"""
        prompt = (
            "你是一个严格的审稿人。审查以下章节草稿的质量，判断是否需要修改。\n\n"
            "审查标准：\n"
            "1. 信息准确性：数据是否可信，来源是否标注\n"
            "2. 内容完整性：是否覆盖了章节主题的关键方面\n"
            "3. 逻辑连贯性：段落之间是否有清晰的逻辑关系\n"
            "4. 表达清晰度：语言是否简洁明了\n\n"
            "返回格式：严格返回 JSON\n"
            '{"verdict": "accept"|"revise", "notes": "修改意见（revise时必填）"}\n\n'
            f"## 章节主题\n{section}\n\n"
            f"## 草稿内容\n{draft}\n\n"
            "## 审查结果（JSON）"
        )
        try:
            resp = await self.model.ainvoke(prompt)
            text = resp.content.strip()
            if text.startswith("```"):
                text = text.split("\n", 1)[1].rsplit("\n", 1)[0]
                if text.startswith("json"):
                    text = text[4:]
            result = json.loads(text)
            logger.info(f"[Reviewer] {section[:30]}: {result.get('verdict')}")
            return result
        except Exception as e:
            logger.warning(f"[Reviewer] 审查失败: {e}")
            return {"verdict": "accept", "notes": ""}


reviewer_agent = ReviewerAgent()