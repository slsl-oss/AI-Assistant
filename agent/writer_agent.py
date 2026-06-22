"""
写作 Agent：主工作流 writer 节点，撰写引言/结论/目录/来源
"""
from model.factory import research_agent_model
from utils.logger_handler import logger


class WriterAgent:
    """撰写引言、结论、目录、来源"""

    def __init__(self):
        self.model = research_agent_model

    async def run(self, title: str, sections: list[str], research_data: list[dict]) -> dict:
        """主工作流 writer 节点。Returns: {introduction, conclusion, table_of_contents, sources}"""
        sources = []
        seen = set()
        for rd in research_data:
            for src in rd.get("sources", []):
                if src not in seen:
                    sources.append(src)
                    seen.add(src)

        toc = "\n".join([f"{i}. {s}" for i, s in enumerate(sections, 1)])
        introduction = await self._write_introduction(title, sections, research_data)
        conclusion = await self._write_conclusion(title, research_data)

        logger.info(f"[Writer] 引言{len(introduction)}字, 结论{len(conclusion)}字, {len(sources)}来源")
        return {"introduction": introduction, "conclusion": conclusion, "table_of_contents": toc, "sources": sources}

    async def _write_introduction(self, title: str, sections: list[str], research_data: list[dict]) -> str:
        context = "\n".join([rd.get("draft", "")[:300] for rd in research_data[:2]])
        prompt = (
            "为以下研究报告撰写引言。200-400 字，概述背景、目的和结构，不使用 emoji。\n\n"
            f"标题：{title}\n章节：{', '.join(sections)}\n研究摘要：{context[:500]}\n\n## 引言"
        )
        try:
            resp = await self.model.ainvoke(prompt)
            return resp.content.strip()
        except Exception as e:
            logger.error(f"[Writer] 引言失败: {e}")
            return ""

    async def _write_conclusion(self, title: str, research_data: list[dict]) -> str:
        summaries = "\n".join([f"- {rd['section']}: {rd['draft'][:200]}" for rd in research_data])
        prompt = (
            "为以下研究报告撰写结论。200-400 字，总结关键发现，给出展望，不使用 emoji。\n\n"
            f"标题：{title}\n各章节摘要：\n{summaries[:2000]}\n\n## 结论"
        )
        try:
            resp = await self.model.ainvoke(prompt)
            return resp.content.strip()
        except Exception as e:
            logger.error(f"[Writer] 结论失败: {e}")
            return ""


writer_agent = WriterAgent()