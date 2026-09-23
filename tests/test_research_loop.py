"""Offline contract/golden tests. No API keys or application imports required."""
import asyncio
import copy
import sys
import unittest

from agent.infrastructure.research_demo import DemoBackend, DemoDecisionModel, demo
from agent.infrastructure.research_loop import ResearchLimits, ResearchLoop, research_complete
from agent.research_agent import ResearchAgent
from agent.tools.research_tools import ResearchTools, TOOL_SCHEMAS, public_url


def tool(name="search_knowledge_base", arguments=None, **extra):
    return {"action": "tool", "tool": name, "arguments": arguments or {"query": "内部制度"},
            "gaps": [], "conflicts": [], **extra}


def finish(citations=None, **extra):
    return {"action": "finish", "answer": "有依据的测试结果", "citations": citations or ["e1"],
            "gaps": [], "conflicts": [], **extra}


class ScriptedModel:
    def __init__(self, decisions):
        self.decisions, self.requests = decisions, []

    async def decide(self, request, *, max_output_tokens):
        self.requests.append(copy.deepcopy(request))
        return copy.deepcopy(self.decisions[min(len(self.requests) - 1, len(self.decisions) - 1)])


class LoopTests(unittest.IsolatedAsyncioTestCase):
    def loop(self, decisions, backend=None, **limits):
        return ResearchLoop(ScriptedModel(decisions), ResearchTools(backend or DemoBackend()), ResearchLimits(**limits))

    async def test_conflict_golden_observations_reach_next_decision(self):
        result = (await demo())["research_result"]
        self.assertEqual(result["status"], "completed")
        calls = [e["tool"] for e in result["trace"] if e["event"] == "observation"]
        self.assertEqual(calls, ["search_knowledge_base", "search_web", "fetch_page", "fetch_page", "search_web", "fetch_page"])
        self.assertEqual(result["usage"]["tool_calls"], 6)
        self.assertTrue(any("120 与 80" in " ".join(m["decision"]["conflicts"])
                            for m in result["history"] if m["role"] == "assistant"))
        self.assertIn("统计口径", result["answer"])
        known = {e["id"] for e in result["evidence"]}
        self.assertTrue(set(result["citations"]) <= known)

    async def test_tool_result_not_just_execution_notification_is_returned(self):
        loop = self.loop([tool(), finish()])
        result = await loop.run("内部制度")
        observation = loop.model.requests[1]["history"][-1]["observation"]
        self.assertIn("记录证据来源", observation["evidence"][0]["text"])
        self.assertEqual(observation["call_id"], "call_1")
        self.assertEqual(result["status"], "completed")

    async def test_knowledge_sufficient_skips_web_and_fetch(self):
        result = (await demo("knowledge"))["research_result"]
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["usage"]["tool_calls"], 1)
        self.assertEqual([e["kind"] for e in result["evidence"]], ["knowledge"])

    async def test_empty_results_request_input_instead_of_report(self):
        result = (await demo("missing"))["research_result"]
        self.assertEqual(result["status"], "needs_input")
        self.assertEqual(result["answer"], "")
        self.assertTrue(result["gaps"])
        self.assertTrue(all(m["observation"]["status"] == "empty" for m in result["history"] if m["role"] == "tool"))

    async def test_call_budget_prevents_dispatch(self):
        result = (await demo("conflict", 2))["research_result"]
        self.assertEqual(result["status"], "budget_exhausted")
        self.assertEqual(result["stop_reason"], "tool_call_budget")
        self.assertEqual(result["usage"]["tool_calls"], 2)
        self.assertEqual(result["answer"], "")

    async def test_step_budget_limits_rejected_actions(self):
        result = await self.loop([tool("shell")], max_steps=3).run("test")
        self.assertEqual(result["stop_reason"], "step_budget")
        self.assertEqual(result["usage"]["model_calls"], 3)
        self.assertEqual(result["usage"]["tool_calls"], 0)

    async def test_token_budget_rejects_before_model(self):
        result = await self.loop([tool()], max_tokens=1).run("test")
        self.assertEqual(result["stop_reason"], "token_budget")
        self.assertEqual(result["usage"]["model_calls"], 0)

    async def test_model_output_size_is_bounded(self):
        result = await self.loop([finish(answer="x" * 1000)], max_output_tokens=100).run("test")
        self.assertEqual(result["stop_reason"], "model_output_budget")

    async def test_name_and_parameters_fail_closed(self):
        attempts = [tool("shell"), tool(["fetch_page"]), tool(arguments={"query": "x", "user_id": "other"}),
                    tool(arguments={"query": ""}), tool(arguments={"query": "x" * 2001}),
                    tool("fetch_page", {"url": "https://example.org/not-discovered"}),
                    tool("fetch_page", {"url": "http://127.0.0.1/secrets"})]
        for decision in attempts:
            with self.subTest(decision=decision):
                # Reserve enough output for the oversize-argument fixture so
                # this test reaches argument validation, not the token guard.
                result = await self.loop([decision], max_steps=1, max_output_tokens=4096).run("test")
                self.assertEqual(result["usage"]["tool_calls"], 0)
                self.assertEqual(result["history"][-1]["observation"]["status"], "rejected")

    async def test_duplicate_tool_call_uses_no_additional_dispatch(self):
        result = await self.loop([tool()], max_steps=3).run("test")
        self.assertEqual(result["usage"]["tool_calls"], 1)
        self.assertEqual(result["history"][-1]["observation"]["error"], "duplicate_call_use_existing_observation")

    async def test_tool_error_returned_and_no_secret_leak(self):
        class Broken(DemoBackend):
            async def search_knowledge_base(self, query):
                raise RuntimeError("secret credential")
        loop = self.loop([tool(), {"action": "need_input", "gaps": ["工具不可用"], "conflicts": []}], Broken())
        result = await loop.run("test")
        self.assertEqual(result["usage"]["tool_calls"], 1)
        self.assertEqual(loop.model.requests[1]["history"][-1]["observation"]["status"], "error")
        self.assertNotIn("secret credential", str(result))

    async def test_snippets_are_not_sufficient_to_finish(self):
        result = await self.loop([tool("search_web", {"query": "latest"}), finish()]).run("test")
        self.assertEqual(result["stop_reason"], "invalid_completion_evidence")

    async def test_forged_citation_no_evidence_gap_and_conflict_block_finish(self):
        for last in [finish(["invented"]), finish(gaps=["缺少日期"]), finish(conflicts=["口径冲突"])]:
            with self.subTest(last=last):
                result = await self.loop([tool(), last]).run("test")
                self.assertEqual(result["status"], "needs_input")
                self.assertEqual(result["answer"], "")
        self.assertEqual((await self.loop([finish()]).run("test"))["status"], "needs_input")

    async def test_malformed_model_outputs_are_explicit_failures(self):
        for decision in [None, [], "not json", {"action": []}, {"action": {}, "gaps": [], "conflicts": []},
                         {"action": "finish"}, {"action": "unknown"}]:
            with self.subTest(decision=decision):
                result = await self.loop([decision]).run("test")
                self.assertEqual(result["status"], "failed")
                self.assertEqual(result["stop_reason"], "invalid_model_output")

    async def test_backend_cannot_forge_evidence_ids_or_redirect(self):
        class Forged(DemoBackend):
            async def search_knowledge_base(self, query):
                return [{"id": "forged", "kind": "page", "title": "t", "text": "evidence"}]
        result = await self.loop([tool(), finish()], Forged()).run("test")
        self.assertEqual(result["evidence"][0]["id"], "e1")
        self.assertEqual(result["evidence"][0]["kind"], "knowledge")

        class Redirected(DemoBackend):
            async def fetch_page(self, url):
                return [{"url": "http://127.0.0.1/", "title": "t", "text": "x"}]
        result = await self.loop([tool("search_web", {"query": "x"}),
                                  tool("fetch_page", {"url": "https://example.org/latest"})], Redirected(), max_steps=2).run("test")
        self.assertEqual(result["history"][-1]["observation"]["status"], "rejected")
        self.assertFalse(any(e["kind"] == "page" for e in result["evidence"]))

    async def test_backend_result_limits_and_truncation(self):
        class TooMany(DemoBackend):
            async def search_knowledge_base(self, query):
                return [{"title": "t", "text": "x"}] * 6
        result = await self.loop([tool()], TooMany(), max_steps=1).run("test")
        self.assertEqual(result["history"][-1]["observation"]["status"], "rejected")
        class Large(DemoBackend):
            async def search_knowledge_base(self, query):
                return [{"title": "t", "text": "x" * 5000}]
        result = await self.loop([tool(), finish()], Large()).run("test")
        self.assertEqual(len(result["evidence"][0]["text"]), 4000)
        self.assertTrue(result["evidence"][0]["truncated"])

    async def test_model_cannot_mutate_runtime_evidence(self):
        class Mutator(ScriptedModel):
            async def decide(self, request, **kwargs):
                if request["history"]:
                    request["history"][-1]["observation"]["evidence"][0]["id"] = "invented"
                return await super().decide(request, **kwargs)
        result = await ResearchLoop(Mutator([tool(), finish(["invented"])]), ResearchTools(DemoBackend())).run("test")
        self.assertEqual(result["status"], "needs_input")
        self.assertEqual(result["evidence"][0]["id"], "e1")

    async def test_model_exception_and_timeout(self):
        class Slow:
            async def decide(self, *args, **kwargs):
                await asyncio.sleep(1)
        result = await ResearchLoop(Slow(), ResearchTools(DemoBackend()), ResearchLimits(call_timeout_seconds=0.01)).run("test")
        self.assertEqual(result["stop_reason"], "model_timeout")
        class Error:
            async def decide(self, *args, **kwargs):
                raise RuntimeError("private")
        result = await ResearchLoop(Error(), ResearchTools(DemoBackend())).run("test")
        self.assertEqual(result["stop_reason"], "model_error")
        self.assertNotIn("private", str(result))

    async def test_total_deadline_and_tool_timeout(self):
        class Slow(DemoBackend):
            async def search_knowledge_base(self, query):
                await asyncio.sleep(1)
        result = await self.loop([tool()], Slow(), max_seconds=0.01).run("test")
        self.assertEqual(result["stop_reason"], "time_budget")
        result = await self.loop([tool()], Slow(), call_timeout_seconds=0.01, max_steps=1).run("test")
        self.assertEqual(result["history"][-1]["observation"]["error"], "tool_timeout")

    async def test_cancel_before_or_during_call(self):
        event = asyncio.Event()
        event.set()
        result = await self.loop([tool()]).run("test", cancel_event=event)
        self.assertEqual(result["status"], "cancelled")
        self.assertEqual(result["usage"]["model_calls"], 0)
        event.clear()
        cleaned = asyncio.Event()
        class Slow:
            async def decide(self, *args, **kwargs):
                event.set()
                try:
                    await asyncio.sleep(1)
                finally:
                    cleaned.set()
        result = await ResearchLoop(Slow(), ResearchTools(DemoBackend())).run("test", cancel_event=event)
        self.assertEqual(result["status"], "cancelled")
        self.assertTrue(cleaned.is_set())

    async def test_concurrent_runs_do_not_share_evidence_or_budget(self):
        loop = ResearchLoop(DemoDecisionModel(), ResearchTools(DemoBackend()))
        complete, empty = await asyncio.gather(loop.run("内部制度"), loop.run("无资料"))
        self.assertEqual(complete["status"], "completed")
        self.assertEqual(empty["status"], "needs_input")
        self.assertFalse(empty["evidence"])

    async def test_agent_legacy_contract_and_offline_string_stream(self):
        class Pipeline:
            closed = False
            async def process(self, topic):
                return {"merged_draft": "old draft", "all_sources": ["https://example.org/"]}
            async def close(self):
                self.closed = True
        pipeline = Pipeline()
        agent = ResearchAgent(pipeline=pipeline)
        self.assertEqual(await agent.run_depth_research("topic"),
                         {"section": "topic", "draft": "old draft", "sources": ["https://example.org/"], "revisions": 0})
        self.assertEqual([s async for s in agent.execute_stream("topic")], ["[研究开始]\n", "old draft"])
        await agent.close()
        self.assertTrue(pipeline.closed)
        lazy = ResearchAgent()
        await lazy.close()
        self.assertIsNone(lazy._pipeline)
        offline = ResearchAgent(research_loop=self.loop([finish()]))
        chunks = [s async for s in offline.execute_stream("topic")]
        self.assertTrue(all(isinstance(s, str) for s in chunks))
        self.assertIn("研究未完成", chunks[-1])
        self.assertIsNone(offline._pipeline)


class GuardTests(unittest.TestCase):
    def test_exactly_three_tools_and_no_scope_parameter(self):
        self.assertEqual({t["name"] for t in TOOL_SCHEMAS}, {"search_web", "search_knowledge_base", "fetch_page"})
        self.assertTrue(all("user_id" not in t["parameters"]["properties"] for t in TOOL_SCHEMAS))

    def test_public_url_syntax(self):
        for url in ["file:///private", "http://localhost/", "http://127.0.0.1/", "http://[::1]/",
                    "http://169.254.169.254/", "http://10.0.0.1/", "https://user:pass@example.org/",
                    "http://2130706433/", "http://example.local/", "https://example.org:9000/"]:
            self.assertFalse(public_url(url), url)
        self.assertTrue(public_url("https://example.org/page"))

    def test_invalid_limits(self):
        for field, value in [("max_steps", 0), ("max_steps", 1.2), ("max_tokens", True), ("max_seconds", float("nan"))]:
            with self.assertRaises(ValueError):
                ResearchLimits(**{field: value})

    def test_completion_gate_checks_expected_sections_and_explicit_status(self):
        self.assertTrue(research_complete(["a"], [{"section": "a"}]))  # legacy compatibility
        for status in ["needs_input", "failed", "budget_exhausted", "cancelled", None]:
            self.assertFalse(research_complete(["a"], [{"section": "a", "research_status": status}]))
        self.assertFalse(research_complete([], []))
        self.assertFalse(research_complete(["a", "b"], [{"section": "a", "research_status": "completed"}]))
        self.assertFalse(research_complete(["a"], [{"section": "other", "research_status": "completed"}]))


if __name__ == "__main__":
    unittest.main()
