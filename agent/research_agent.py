"""
研究型 Agent：提供 browser 节点（初步调研）和深度研究方法
  - run_initial_research: 主工作流 browser 节点，快速初步调研
  - run_depth_research: 子工作流 researcher 节点，三级并行深度研究
"""
import asyncio

from agent.infrastructure.research_pipeline import ResearchPipeline
from utils.logger_handler import logger


class ResearchAgent:
    """研究型 Agent：初步调研 + 深度研究"""

    def __init__(self):
        self.pipeline = ResearchPipeline()

    async def run_initial_research(self, query: str) -> str:
        """主工作流 browser 节点：快速初步调研，返回简要概述"""
        prompt = (
            "你是一个快速调研助手。对以下主题进行简要初步调研，"
            "提取关键概念、主要方向和重要背景信息。\n\n"
            "要求：200-400 字，列出 3-5 个关键点，不使用 emoji\n\n"
            f"主题：{query}"
        )
        try:
            resp = await self.pipeline.model.ainvoke(prompt)
            result = resp.content.strip()
            logger.info(f"[ResearchAgent] 初步调研完成: {len(result)} 字符")
            return result
        except Exception as e:
            logger.error(f"[ResearchAgent] 初步调研失败: {e}")
            return ""

    async def run_depth_research(self, topic: str) -> dict:
        """子工作流 researcher 节点：三级并行深度研究，返回结构化草稿"""
        logger.info(f"[ResearchAgent] 深度研究: {topic[:50]}")
        result = await self.pipeline.process(topic)
        return {
            "section": topic,
            "draft": result["merged_draft"],
            "sources": result.get("all_sources", []),
            "revisions": 0,
        }

    # ==================== 兼容旧接口 ====================

    async def execute_stream(self, query: str, session_id: str = ""):
        """流式执行（兼容旧接口）"""
        try:
            yield "[研究开始]\n"
            result = await self.pipeline.process(query)
            yield result["merged_draft"]
        except Exception as e:
            logger.error(f"[ResearchAgent] 执行失败: {e}")
            yield f"\n研究过程出错: {str(e)}"

    async def run_agent(self, query: str):
        chunks = []
        async for chunk in self.execute_stream(query):
            print(chunk, end="", flush=True)
            chunks.append(chunk)
        return "".join(chunks)

    async def close(self):
        await self.pipeline.close()


research_agent = ResearchAgent()