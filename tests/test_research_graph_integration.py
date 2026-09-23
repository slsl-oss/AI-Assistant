"""Real LangGraph execution, with ONLY external model/DB/legacy agents replaced.

Optional dependency: pip install -r tests/requirements-graph.txt
No configuration, key, network backend, server or database is loaded.
"""
import importlib.util
import logging
import os
from pathlib import Path
import socket
import sys
import types
import unittest
from contextlib import contextmanager
from unittest.mock import AsyncMock, Mock, patch

from agent.infrastructure.research_demo import DemoBackend, DemoDecisionModel
from agent.infrastructure.research_loop import ResearchLimits, ResearchLoop
from agent.research_agent import ResearchAgent
from agent.tools.research_tools import ResearchTools

HAS_LANGGRAPH = importlib.util.find_spec("langgraph") is not None
ROOT = Path(__file__).resolve().parents[1]


def module(**values):
    result = types.ModuleType("offline_fixture")
    result.__dict__.update(values)
    return result


def load_source(relative, name):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


@contextmanager
def offline_graph(researcher, sections):
    planner_model = types.SimpleNamespace(ainvoke=AsyncMock(return_value=types.SimpleNamespace(
        content=__import__("json").dumps({"title": "Offline report", "sections": sections}))))
    reviewer = types.SimpleNamespace(review=AsyncMock(return_value={"verdict": "accept"}))
    reviser = types.SimpleNamespace(revise=AsyncMock(return_value="revised"))
    writer = types.SimpleNamespace(run=AsyncMock(return_value={"introduction": "intro", "conclusion": "end",
                                                              "table_of_contents": "toc", "sources": []}))
    publisher = types.SimpleNamespace(run=AsyncMock(return_value="offline report"))
    store = Mock()
    # Guard even accidental application imports from opening a real connection.
    fakes = {
        "model.factory": module(research_agent_model=planner_model, supervisor_model=planner_model, summarizer_model=planner_model),
        "utils.logger_handler": module(logger=logging.getLogger("offline-test")),
        "utils.config_handler": module(agent_conf={}),
        "utils.postgres_checkpointer": module(AsyncPostgresSaver=Mock()),
        "utils.observability": module(TraceContext=Mock(), set_current_trace=Mock(), clear_current_trace=Mock(), log_trace_summary=Mock()),
        "agent.reviewer_agent": module(reviewer_agent=reviewer),
        "agent.reviser_agent": module(reviser_agent=reviser),
        "agent.react_agent": module(react_agent=Mock()),
        "agent.writer_agent": module(writer_agent=writer),
        "agent.publisher_agent": module(publisher_agent=publisher),
        "agent.human_agent": module(human_agent=types.SimpleNamespace(review_plan=AsyncMock(return_value={"approved": True}))),
        "rag.vector_stores": module(VectorStoreService=Mock(return_value=store)),
    }
    with patch.dict(os.environ, {"LANGSMITH_TRACING": "false", "LANGCHAIN_TRACING_V2": "false"}), patch.dict(sys.modules, fakes):
        editor_module = load_source("agent/editor_agent.py", "offline_editor")
        editor_module.editor_agent = editor_module.EditorAgent(researcher=researcher)
        with patch.dict(sys.modules, {"agent.editor_agent": editor_module}), \
             patch.object(socket.socket, "connect", side_effect=AssertionError("network disabled in graph tests")):
            supervisor_module = load_source("agent/supervisor_agent.py", "offline_supervisor")
            # This fixture replaces the initial-research LLM only, not graph nodes.
            supervisor_module.research_agent = types.SimpleNamespace(run_initial_research=AsyncMock(return_value="offline background"))
            yield supervisor_module, editor_module, reviewer, reviser, writer, publisher, store


@unittest.skipUnless(HAS_LANGGRAPH, "optional LangGraph not installed; run with tests/requirements-graph.txt")
class GraphTests(unittest.IsolatedAsyncioTestCase):
    def researcher(self, **limits):
        return ResearchAgent(research_loop=ResearchLoop(DemoDecisionModel(), ResearchTools(DemoBackend()), ResearchLimits(**limits)))

    async def test_completed_chapter_runs_actual_two_graphs(self):
        sections = ["内部制度"]
        with offline_graph(self.researcher(), sections) as (s, e, review, revise, writer, publisher, store):
            result = await s.supervisor_agent._graph.ainvoke({"task": {"query": "内部制度"}, "plan_revision_count": 0},
                                                            {"configurable": {"thread_id": "complete"}})
            self.assertEqual(result["research_status"], "completed")
            self.assertEqual(result["report"], "offline report")
            self.assertEqual(result["research_data"][0]["research_result"]["evidence"][0]["kind"], "knowledge")
            review.review.assert_awaited_once()
            writer.run.assert_awaited_once()
            publisher.run.assert_awaited_once()
            store.add_report_to_store.assert_called_once()

    async def test_missing_or_budget_exhausted_never_reviews_writes_or_publishes(self):
        for section, limits, expected in [("无资料", {}, "needs_input"),
                                           ("最新资料", {"max_tool_calls": 1}, "budget_exhausted")]:
            with self.subTest(expected=expected), offline_graph(self.researcher(**limits), [section]) as (s, e, review, revise, writer, publisher, store):
                result = await s.supervisor_agent._graph.ainvoke({"task": {"query": section}, "report": "stale report"},
                                                                {"configurable": {"thread_id": expected}})
                self.assertEqual(result["research_status"], "incomplete")
                self.assertEqual(result["research_data"][0]["research_status"], expected)
                self.assertIn("研究未完成", result["report"])
                self.assertNotIn("stale report", result["report"])
                review.review.assert_not_awaited()
                revise.revise.assert_not_awaited()
                writer.run.assert_not_awaited()
                publisher.run.assert_not_awaited()
                store.add_report_to_store.assert_not_called()

    async def test_one_incomplete_parallel_chapter_blocks_whole_report(self):
        with offline_graph(self.researcher(), ["内部制度", "无资料"]) as (s, e, review, revise, writer, publisher, store):
            result = await s.supervisor_agent._graph.ainvoke({"task": {"query": "mixed"}},
                                                            {"configurable": {"thread_id": "mixed"}})
            self.assertEqual([r["research_status"] for r in result["research_data"]], ["completed", "needs_input"])
            self.assertEqual(result["research_status"], "incomplete")
            writer.run.assert_not_awaited()
            publisher.run.assert_not_awaited()

    async def test_direct_writer_and_publisher_calls_are_guarded(self):
        with offline_graph(self.researcher(), ["x"]) as (s, e, review, revise, writer, publisher, store):
            state = {"sections": ["x"], "research_data": [{"section": "x", "research_status": "failed"}]}
            for handler in [s.writer_node, s.publisher_node]:
                with self.assertRaises(ValueError):
                    await handler(state)
            writer.run.assert_not_awaited()
            publisher.run.assert_not_awaited()

    async def test_legacy_researcher_still_passes_graph(self):
        old = types.SimpleNamespace(run_depth_research=AsyncMock(return_value={
            "section": "legacy", "draft": "old draft", "sources": ["https://example.org/"], "revisions": 0}))
        with offline_graph(old, ["legacy"]) as (s, e, review, revise, writer, publisher, store):
            result = await s.supervisor_agent._graph.ainvoke({"task": {"query": "legacy"}},
                                                            {"configurable": {"thread_id": "legacy"}})
            self.assertEqual(result["report"], "offline report")
            self.assertEqual(result["research_data"][0]["sources"], ["https://example.org/"])
            writer.run.assert_awaited_once()

    async def test_incomplete_result_is_visible_in_existing_string_stream(self):
        from utils.cancellation import CancellationToken
        memory = types.SimpleNamespace(search=lambda *a, **k: [], format_memories_for_prompt=lambda value: "")
        with offline_graph(self.researcher(), ["无资料"]) as (s, e, review, revise, writer, publisher, store), \
             patch.dict(sys.modules, {"memory.mem0_service": module(mem0_service=memory)}):
            supervisor = s.supervisor_agent
            supervisor._ensure_context_budget = AsyncMock()
            supervisor._maybe_batch_extract = AsyncMock()
            chunks = [chunk async for chunk in supervisor._execute_storm(
                "无资料", "stream-test", "fixture-user", CancellationToken(), Mock())]
            self.assertTrue(all(isinstance(chunk, str) for chunk in chunks))
            text = "".join(chunks)
            self.assertIn("未完成: needs_input", text)
            self.assertIn("研究未完成，已停止后续写作与发布", text)
            self.assertNotIn("[无资料] 完成", text)
            writer.run.assert_not_awaited()
            publisher.run.assert_not_awaited()


if __name__ == "__main__":
    unittest.main()
