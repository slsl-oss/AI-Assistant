"""
子编排 Agent：planner 节点 + 子编排（researcher → reviewer → reviser 循环）
主工作流节点：
  - plan_research: planner 节点，规划大纲
  - run_parallel_research: researcher 节点，并行研究各章节
子工作流：每个章节独立运行 DraftState
"""
import asyncio
import json
from datetime import datetime
from typing import TypedDict, Optional

from langgraph.graph import StateGraph, END
from model.factory import research_agent_model
from agent.research_agent import research_agent
from agent.reviewer_agent import reviewer_agent
from agent.reviser_agent import reviser_agent
from utils.logger_handler import logger


class DraftState(TypedDict):
    section: str
    draft: str
    review: Optional[str]
    revision_count: int


class EditorAgent:
    """子编排：planner + 子工作流"""

    MAX_REVISIONS = 3

    def __init__(self):
        self.model = research_agent_model
        self._graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        wf = StateGraph(DraftState)
        wf.add_node("researcher", self._researcher_node)
        wf.add_node("reviewer", self._reviewer_node)
        wf.add_node("reviser", self._reviser_node)
        wf.set_entry_point("researcher")
        wf.add_edge("researcher", "reviewer")
        wf.add_edge("reviser", "reviewer")
        wf.add_conditional_edges(
            "reviewer", self._route_after_review,
            {"accept": END, "revise": "reviser"},
        )
        return wf.compile()

    async def _researcher_node(self, state: DraftState) -> dict:
        section = state["section"]
        logger.info(f"[Editor] 研究章节: {section[:40]}")
        result = await research_agent.run_depth_research(section)
        return {"draft": result["draft"], "revision_count": 0, "review": None}

    async def _reviewer_node(self, state: DraftState) -> dict:
        count = state.get("revision_count", 0)
        if count >= self.MAX_REVISIONS:
            logger.info(f"[Editor] 已达最大修改次数，强制 accept")
            return {"review": None}
        result = await reviewer_agent.review(state["section"], state["draft"])
        return {"review": None} if result["verdict"] == "accept" else {"review": result.get("notes", "需要进一步修改")}

    async def _reviser_node(self, state: DraftState) -> dict:
        revised = await reviser_agent.revise(state["section"], state["draft"], state["review"])
        return {"draft": revised, "revision_count": state.get("revision_count", 0) + 1}

    def _route_after_review(self, state: DraftState) -> str:
        return "accept" if state.get("review") is None else "revise"

    # ==================== 主工作流节点 ====================

    async def plan_research(self, query: str, initial_research: str) -> dict:
        """主工作流 planner 节点：规划报告大纲。Returns: {title, sections, headers, date}"""
        prompt = (
            "你是一个专业的报告规划助手。根据用户问题和初步调研结果，规划研究报告大纲。\n\n"
            "返回格式：严格返回 JSON 对象\n"
            '{"title": "报告标题", "sections": ["章节1", "章节2", ...]}\n\n'
            f"用户问题：{query}\n\n"
            f"初步调研：{initial_research or '无'}\n\n"
            "大纲（JSON）："
        )
        try:
            resp = await self.model.ainvoke(prompt)
            text = resp.content.strip()
            if text.startswith("```"):
                text = text.split("\n", 1)[1].rsplit("\n", 1)[0]
                if text.startswith("json"):
                    text = text[4:]
            plan = json.loads(text)
            sections = plan.get("sections", [query])
            title = plan.get("title", query[:50])
            date_str = datetime.now().strftime("%Y年%m月%d日")
            logger.info(f"[Editor] planner: {title}, {len(sections)} 章节")
            return {"title": title, "sections": sections, "headers": {s: s for s in sections}, "date": date_str}
        except Exception as e:
            logger.warning(f"[Editor] planner 失败: {e}")
            return {"title": query[:50], "sections": [query], "headers": {query: query}, "date": datetime.now().strftime("%Y年%m月%d日")}

    async def run_parallel_research(self, sections: list[str]) -> list[dict]:
        """主工作流 researcher 节点：并行研究所有章节，每个章节运行子工作流"""
        logger.info(f"[Editor] 并行研究 {len(sections)} 个章节")

        async def _run_one(section: str) -> dict:
            initial = {"section": section, "draft": "", "review": None, "revision_count": 0}
            result = await self._graph.ainvoke(initial)
            return {"section": section, "draft": result["draft"], "revisions": result.get("revision_count", 0)}

        return await asyncio.gather(*[_run_one(s) for s in sections])


editor_agent = EditorAgent()