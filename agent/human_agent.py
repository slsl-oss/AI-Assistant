"""
人工审批 Agent：在主工作流中提供人工审批节点
"""
from utils.logger_handler import logger


class HumanAgent:
    """人工审批：检查大纲是否需要修改"""

    async def review_plan(
        self, query: str, sections: list[str], initial_research: str = ""
    ) -> dict:
        """审查大纲规划。当前自动通过，后续可接入人工审批界面。"""
        logger.info(f"[Human] 自动审批大纲: {len(sections)} 个章节")
        return {"approved": True, "feedback": ""}

    async def review_final(self, report: str) -> dict:
        """审查最终报告"""
        logger.info(f"[Human] 自动审批最终报告: {len(report)} 字符")
        return {"approved": True, "feedback": ""}


human_agent = HumanAgent()