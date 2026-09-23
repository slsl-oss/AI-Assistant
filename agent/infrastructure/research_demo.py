"""Run: python -B -m agent.infrastructure.research_demo --scenario conflict

All data and decisions below are synthetic. This is an observation-driven mock
policy for exercising the loop, NOT an LLM, benchmark or real research result.
"""
import argparse
import asyncio
import json

from agent.infrastructure.research_loop import ResearchLimits, ResearchLoop
from agent.research_agent import ResearchAgent
from agent.tools.research_tools import ResearchTools


class DemoBackend:
    async def search_knowledge_base(self, query):
        if "内部制度" in query:
            return [{"title": "模拟内部制度", "text": "模拟制度：研究报告需要记录证据来源。"}]
        if "无资料" in query:
            return []
        return [{"title": "模拟历史数据", "text": "示例产品 2024 年用户数为 60。",
                 "published_at": "2024-12-31"}]

    async def search_web(self, query):
        if "无资料" in query:
            return []
        if "口径" in query:
            return [{"title": "模拟官方口径说明", "url": "https://example.org/correction",
                     "text": "官方发布统计口径说明，需读取正文。", "published_at": "2026-09-02"}]
        return [
            {"title": "模拟最新公告", "url": "https://example.org/latest", "text": "2026 年为 120，详见正文。",
             "published_at": "2026-09-01"},
            {"title": "模拟二手报道", "url": "https://example.net/news", "text": "报道称当前为 80，口径待核实。",
             "published_at": "2026-09-01"},
        ]

    async def fetch_page(self, url):
        pages = {
            "https://example.org/latest": "模拟公告原文：2026 年注册用户数为 120，未说明活跃口径。",
            "https://example.net/news": "模拟报道原文：2026 年活跃用户数为 80，报道标题没有区分口径。",
            "https://example.org/correction": "模拟官方说明：2026 年注册用户数为 120，活跃用户数为 80；二者口径不同。",
        }
        return [{"title": "模拟网页正文", "text": pages[url], "url": url, "published_at": "2026-09-02"}]


class DemoDecisionModel:
    """Chooses from observed evidence, never from an unconditional action queue."""
    async def decide(self, request, *, max_output_tokens):
        observations = [m["observation"] for m in request["history"] if m["role"] == "tool"]
        evidence = [e for o in observations for e in o["evidence"]]
        base = {"gaps": [], "conflicts": []}

        def call(name, arguments, reason, conflicts=None):
            return {"action": "tool", "tool": name, "arguments": arguments, "reason": reason,
                    "gaps": [reason], "conflicts": conflicts or []}

        if not observations:
            return call("search_knowledge_base", {"query": request["question"]}, "先检查内部资料是否足够")
        kb = [e for e in evidence if e["kind"] == "knowledge"]
        if kb and "模拟制度" in kb[0]["text"]:
            return {**base, "action": "finish", "answer": kb[0]["text"], "citations": [kb[0]["id"]]}
        if not any(o["tool"] == "search_web" for o in observations):
            return call("search_web", {"query": request["question"] + " 最新数据"}, "知识库没有当前数据，联网补充")
        snippets = [e for e in evidence if e["kind"] == "snippet"]
        pages = [e for e in evidence if e["kind"] == "page"]
        if not snippets:
            return {"action": "need_input", "gaps": ["没有找到资料，请提供可公开检索的标题或可信来源"], "conflicts": []}
        # A failed/blocked fetch is not silently converted into an answer.
        if observations[-1]["status"] in {"error", "rejected"}:
            return {"action": "need_input", "gaps": ["工具执行未成功，请核实来源或稍后重试"], "conflicts": []}
        fetched = {e["url"] for e in pages}
        for snippet in snippets:
            if snippet["url"] not in fetched:
                return call("fetch_page", {"url": snippet["url"]}, "摘要不够，读取原文核实",
                            ["注册用户与活跃用户口径需要核实"] if len(pages) >= 2 else [])
        correction = next((e for e in pages if "二者口径不同" in e["text"]), None)
        if correction is None:
            return call("search_web", {"query": "示例产品 2026 官方统计口径说明"}, "发现不同来源数据不一致，补查官方口径",
                        ["120 与 80 的统计口径需独立说明"])
        return {**base, "action": "finish", "citations": [e["id"] for e in pages],
                "answer": "【离线模拟】注册用户 120、活跃用户 80，差异来自统计口径，不能把二者当作同一指标。"}


async def demo(scenario="conflict", max_tool_calls=8):
    question = {"conflict": "示例产品 2026 年最新用户数是多少？",
                "knowledge": "查询内部制度：研究报告需要记录什么？", "missing": "无资料：未知主题"}[scenario]
    agent = ResearchAgent(research_loop=ResearchLoop(
        DemoDecisionModel(), ResearchTools(DemoBackend()), ResearchLimits(max_tool_calls=max_tool_calls)))
    return await agent.run_depth_research(question)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", choices=["conflict", "knowledge", "missing"], default="conflict")
    parser.add_argument("--max-tool-calls", type=int, default=8)
    args = parser.parse_args()
    result = asyncio.run(demo(args.scenario, args.max_tool_calls))
    detail = result["research_result"]
    print(json.dumps({"simulation": True, "status": result["research_status"], "draft": result["draft"],
                      "stop_reason": detail["stop_reason"], "gaps": detail["gaps"], "conflicts": detail["conflicts"],
                      "trace": detail["trace"], "usage": detail["usage"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
