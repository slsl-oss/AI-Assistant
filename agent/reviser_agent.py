"""
修改 Agent：根据审查意见修改草稿
"""
from model.factory import research_agent_model
from utils.logger_handler import logger


class ReviserAgent:
    """根据审查意见修改草稿"""

    def __init__(self):
        self.model = research_agent_model

    async def revise(self, section: str, draft: str, notes: str) -> str:
        """根据审查意见修改草稿，返回修改后的版本"""
        prompt = (
            "你是一个专业的编辑。根据审稿人的修改意见，修订以下章节草稿。\n\n"
            "要求：\n"
            "- 逐条对照修改意见进行调整\n"
            "- 保留原文中正确的部分，不要全盘重写\n"
            "- 保留所有来源引用\n"
            "- 不使用 emoji 表情符号\n\n"
            f"## 章节主题\n{section}\n\n"
            f"## 原草稿\n{draft}\n\n"
            f"## 修改意见\n{notes}\n\n"
            "## 修订稿"
        )
        try:
            resp = await self.model.ainvoke(prompt)
            result = resp.content.strip()
            logger.info(f"[Reviser] {section[:30]}: 修改完成")
            return result
        except Exception as e:
            logger.error(f"[Reviser] 修改失败: {e}")
            return draft


reviser_agent = ReviserAgent()