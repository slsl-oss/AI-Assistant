"""
ChiefEditor Supervisor：STORM 主工作流编排
  browser → planner → human → researcher → writer → publisher

简单查询 → react_agent 直接处理（不进入图）
深度研究 → ResearchState LangGraph 主工作流
"""
import asyncio
import operator
from datetime import datetime
from typing import TypedDict, Annotated, List

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage

from model.factory import supervisor_model, summarizer_model
from utils.logger_handler import logger
from utils.cancellation import CancellationToken, TaskCancelledError
from utils.observability import TraceContext, set_current_trace, clear_current_trace, log_trace_summary

from agent.react_agent import react_agent
from agent.research_agent import research_agent
from agent.editor_agent import editor_agent
from agent.writer_agent import writer_agent
from agent.publisher_agent import publisher_agent
from agent.human_agent import human_agent

CANCELLED_SIGNAL = object()


# ==================== 主工作流 State ====================

class ResearchState(TypedDict):
    task: dict
    initial_research: str
    sections: List[str]
    research_data: List[dict]
    human_feedback: str
    plan_revision_count: int
    title: str
    headers: dict
    date: str
    table_of_contents: str
    introduction: str
    conclusion: str
    sources: List[str]
    report: str
    messages: Annotated[List[BaseMessage], operator.add]
    session_id: str
    user_id: str


# ==================== 节点函数 ====================

async def browser_node(state: ResearchState) -> dict:
    """browser: 初步调研 → initial_research"""
    query = state["task"].get("query", "")
    logger.info(f"[Browser] {query[:50]}")
    initial = await research_agent.run_initial_research(query)
    return {"initial_research": initial}


async def planner_node(state: ResearchState) -> dict:
    """planner: 规划大纲 → title, sections, headers, date"""
    query = state["task"].get("query", "")
    plan = await editor_agent.plan_research(query, state.get("initial_research", ""))
    logger.info(f"[Planner] {plan['title']}, {len(plan['sections'])} 章节")
    return plan


async def human_node(state: ResearchState) -> dict:
    """human: 人工审批 → human_feedback, plan_revision_count"""
    count = state.get("plan_revision_count", 0) + 1
    result = await human_agent.review_plan(
        state["task"].get("query", ""), state.get("sections", []),
        state.get("initial_research", "")
    )
    logger.info(f"[Human] {'通过' if result['approved'] else '驳回'} (第{count}次)")
    if result["approved"]:
        return {"human_feedback": "", "plan_revision_count": count}
    return {"human_feedback": result.get("feedback", "请修改"), "plan_revision_count": count}


def route_after_human(state: ResearchState) -> str:
    return "planner" if state.get("human_feedback") else "researcher"


async def researcher_node(state: ResearchState) -> dict:
    """researcher: 并行研究各章节 → research_data"""
    sections = state.get("sections", [])
    logger.info(f"[Researcher] {len(sections)} 章节并行")
    data = await editor_agent.run_parallel_research(sections)
    return {"research_data": data}


async def writer_node(state: ResearchState) -> dict:
    """writer: 引言/结论/目录/来源"""
    result = await writer_agent.run(
        state.get("title", ""), state.get("sections", []),
        state.get("research_data", [])
    )
    logger.info(f"[Writer] 引言{len(result['introduction'])}字, 结论{len(result['conclusion'])}字")
    return result


async def publisher_node(state: ResearchState) -> dict:
    """publisher: 排版导出 → report"""
    report = await publisher_agent.run(
        title=state.get("title", ""),
        date=state.get("date", datetime.now().strftime("%Y年%m月%d日")),
        introduction=state.get("introduction", ""),
        conclusion=state.get("conclusion", ""),
        table_of_contents=state.get("table_of_contents", ""),
        research_data=state.get("research_data", []),
        sources=state.get("sources", []),
        headers=state.get("headers", {}),
    )
    logger.info(f"[Publisher] {len(report)} 字符")

    # 存入向量数据库，供后续 RAG 检索
    try:
        from rag.vector_stores import VectorStoreService
        vs = VectorStoreService()
        vs.add_report_to_store(
            title=state.get("title", ""), content=report,
            user_id=state.get("user_id", "default_user"),
        )
    except Exception as e:
        logger.warning(f"[Publisher] 报告存入向量库失败: {e}")

    return {"report": report, "messages": [AIMessage(content=report)]}


# ==================== SupervisorAgent ====================

class SupervisorAgent:
    """ChiefEditor：STORM 主工作流编排"""

    def __init__(self):
        self._checkpointer = self._init_checkpointer()
        self._graph = self._build_graph()
        self._active_tokens: dict[str, CancellationToken] = {}

    def _init_checkpointer(self):
        try:
            from utils.postgres_checkpointer import AsyncPostgresSaver
            from utils.config_handler import agent_conf
            db_uri = agent_conf.get("DB_URI")
            if db_uri:
                saver = AsyncPostgresSaver.from_conn_string(db_uri)
                saver.setup()
                return saver
        except Exception as e:
            logger.warning(f"PostgreSQL连接失败，使用内存存储: {e}")
        return MemorySaver()

    def _build_graph(self) -> StateGraph:
        """browser → planner → human → researcher → writer → publisher"""
        wf = StateGraph(ResearchState)
        wf.add_node("browser", browser_node)
        wf.add_node("planner", planner_node)
        wf.add_node("human", human_node)
        wf.add_node("researcher", researcher_node)
        wf.add_node("writer", writer_node)
        wf.add_node("publisher", publisher_node)

        wf.set_entry_point("browser")
        wf.add_edge("browser", "planner")
        wf.add_edge("planner", "human")
        wf.add_conditional_edges("human", route_after_human, {
            "planner": "planner", "researcher": "researcher",
        })
        wf.add_edge("researcher", "writer")
        wf.add_edge("writer", "publisher")
        wf.add_edge("publisher", END)

        return wf.compile(checkpointer=self._checkpointer)

    # ==================== 执行入口 ====================

    async def execute_stream(self, query: str, session_id: str = "", user_id: str = "default_user"):
        if session_id in self._active_tokens:
            yield "该会话已有活跃任务"
            return

        token = CancellationToken()
        self._active_tokens[session_id] = token

        # 创建追踪上下文
        trace = TraceContext(
            session_id=session_id, user_id=user_id, query=query,
            model="deepseek-v4-pro",
        )
        set_current_trace(trace)

        try:
            trace.start_step("classify")
            is_deep = await self._is_deep_research(query)
            trace.end_step()
            trace.is_deep_research = is_deep

            if not is_deep:
                trace.agent_path = "react_agent"
                async for chunk in self._execute_simple(query, session_id, user_id, token, trace):
                    yield chunk
            else:
                trace.agent_path = "STORM(browser→planner→human→researcher→writer→publisher)"
                async for chunk in self._execute_storm(query, session_id, user_id, token, trace):
                    yield chunk
        except Exception as e:
            import traceback
            trace.add_error("execute_stream", str(e))
            logger.error(f"[ExecuteStream] {traceback.format_exc()}")
            yield f"执行失败: {str(e)}"
        finally:
            trace.finish()
            log_trace_summary(trace)
            clear_current_trace()
            self._active_tokens.pop(session_id, None)

    async def _execute_simple(self, query: str, session_id: str, user_id: str, token: CancellationToken, trace: TraceContext):
        trace.start_step("react_agent")
        from memory.mem0_service import mem0_service
        mems = mem0_service.search(query, user_id=user_id)
        ctx = mem0_service.format_memories_for_prompt(mems)
        aq = f"{ctx}\n\n[用户当前问题]\n{query}" if ctx else query
        await self._ensure_context_budget(session_id, user_id)
        output = ""
        async for chunk in react_agent.execute_stream(aq, session_id):
            if token.is_cancelled:
                yield CANCELLED_SIGNAL
                trace.end_step(success=False, error="cancelled")
                return
            output += chunk
            yield chunk
        trace.add_output_tokens(output)
        trace.end_step()
        await self._maybe_batch_extract(session_id, user_id)

    async def _execute_storm(self, query: str, session_id: str, user_id: str, token: CancellationToken, trace: TraceContext):
        from memory.mem0_service import mem0_service
        mems = mem0_service.search(query, user_id=user_id)
        ctx = mem0_service.format_memories_for_prompt(mems)
        aq = f"{ctx}\n\n[用户当前问题]\n{query}" if ctx else query
        await self._ensure_context_budget(session_id, user_id)

        initial = {
            "task": {"query": query}, "initial_research": "", "sections": [],
            "research_data": [], "human_feedback": "", "plan_revision_count": 0,
            "title": "", "headers": {}, "date": "", "table_of_contents": "",
            "introduction": "", "conclusion": "", "sources": [], "report": "",
            "messages": [HumanMessage(content=aq)],
            "session_id": session_id, "user_id": user_id,
        }
        config = {"configurable": {"thread_id": session_id, "user_id": user_id}}

        labels = {
            "browser": "[Browser] 初步调研", "planner": "[Planner] 规划大纲",
            "human": "[Human] 人工审批", "researcher": "[Researcher] 并行研究",
            "writer": "[Writer] 撰写报告", "publisher": "[Publisher] 排版导出",
        }

        try:
            async for ev in self._graph.astream_events(initial, config, version="v2"):
                kind = ev["event"]
                if kind == "on_chain_start":
                    name = ev.get("name", "")
                    if name in labels:
                        trace.start_step(name)
                        yield f"\n{labels[name]}...\n"
                elif kind == "on_chain_end":
                    name = ev.get("name", "")
                    output = ev.get("data", {}).get("output", {})
                    if name == "browser":
                        trace.end_step()
                        yield f"  调研完成 ({len(str(output.get('initial_research', '')))} 字符)\n"
                    elif name == "planner":
                        trace.end_step()
                        secs = output.get("sections", [])
                        yield f"  规划 {len(secs)} 个章节\n"
                    elif name == "researcher":
                        trace.end_step()
                        for r in output.get("research_data", []):
                            yield f"  [{r['section'][:30]}] 完成 (修改{r.get('revisions',0)}次)\n"
                    elif name == "writer":
                        trace.end_step()
                        yield f"  引言/结论/目录完成\n"
                    elif name == "publisher":
                        report = output.get("report", "")
                        if report:
                            trace.add_output_tokens(report)
                        trace.end_step()
                        if report:
                            yield f"\n{report}"
            await self._maybe_batch_extract(session_id, user_id)
        except asyncio.CancelledError:
            token.cancel()
            yield CANCELLED_SIGNAL

    async def _is_deep_research(self, query: str) -> bool:
        prompt = (
            "判断用户问题是否需要深度研究报告（多章节、结构化、引用来源）。"
            "简单问答、日期天气、闲聊 → false。"
            "研究报告、深度分析、行业调研、技术综述 → true。\n\n"
            "返回：只返回 true 或 false\n\n"
            f"问题：{query}"
        )
        try:
            resp = await supervisor_model.ainvoke(prompt)
            return "true" in resp.content.strip().lower()
        except Exception:
            return False

    # ==================== 辅助方法 ====================

    async def run_agent(self, query: str, session_id: str = "") -> str:
        r = []
        async for c in self.execute_stream(query, session_id):
            if c is CANCELLED_SIGNAL:
                raise TaskCancelledError(f"Session {session_id} was cancelled")
            r.append(c)
            print(c, end="", flush=True)
        return "".join(r)

    def cancel_session(self, session_id: str) -> bool:
        token = self._active_tokens.get(session_id)
        if token:
            token.cancel()
            return True
        return False

    async def _ensure_context_budget(self, session_id: str, user_id: str):
        TOKEN_BUDGET = 20000
        config = {"configurable": {"thread_id": session_id, "user_id": user_id}}
        ck = await self._checkpointer.aget_tuple(config)
        if not ck or not ck.checkpoint:
            return
        msgs = ck.checkpoint.get("channel_values", {}).get("messages", [])
        if not msgs:
            return
        pairs = []
        um = None
        um_obj = None
        for m in msgs:
            if hasattr(m, "type") and m.type == "human" and m.content:
                um = m.content
                um_obj = m
            elif hasattr(m, "type") and m.type == "ai" and m.content and um:
                pairs.append((um, m.content, um_obj, m))
                um = None
                um_obj = None

        def _est(t):
            zh = sum(1 for c in t if '一' <= c <= '鿿')
            return int(zh * 1.5 + (len(t) - zh) * 0.25)

        total = 0
        cutoff = 0
        for i in range(len(pairs) - 1, -1, -1):
            u, a, _, _ = pairs[i]
            total += _est(u[:500]) + _est(a[:500])
            if total > TOKEN_BUDGET:
                cutoff = i + 1
                break
        if cutoff == 0:
            return

        old_summary = ""
        if hasattr(msgs[0], "type") and msgs[0].type == "system":
            old_summary = msgs[0].content
        text = (f"[前序摘要]\n{old_summary}\n\n" if old_summary else "")
        for u, a, _, _ in pairs[:cutoff]:
            text += f"用户：{u[:500]}\nAI：{a[:500]}\n"
        try:
            resp = await summarizer_model.ainvoke(
                "总结以下对话的关键信息，保留用户偏好、个人事实、重要决策。用简洁中文。\n\n" + text)
            new_summary = resp.content.strip()
            if not new_summary:
                return
        except Exception:
            return
        from langchain_core.messages import SystemMessage
        new_msgs = [SystemMessage(content=f"[历史对话摘要]\n{new_summary}")]
        for _, _, hm, am in pairs[cutoff:]:
            new_msgs.append(hm)
            new_msgs.append(am)
        ck.checkpoint["channel_values"]["messages"] = new_msgs
        await self._checkpointer.aput(config, ck.checkpoint, ck.metadata,
                                      ck.checkpoint.get("channel_versions", {}))
        logger.info(f"[Compress] session={session_id}, {cutoff}/{len(pairs)}轮→摘要")

    async def _maybe_batch_extract(self, session_id: str, user_id: str):
        try:
            ck = await self._checkpointer.aget_tuple(
                {"configurable": {"thread_id": session_id, "user_id": user_id}})
            if not ck or not ck.checkpoint:
                return
            msgs = ck.checkpoint.get("channel_values", {}).get("messages", [])
            pairs = []
            um = None
            for m in msgs:
                if hasattr(m, "type") and m.type == "human" and m.content:
                    um = m.content
                elif hasattr(m, "type") and m.type == "ai" and m.content and um:
                    pairs.append((um, m.content))
                    um = None
            if len(pairs) < 10:
                return
            last = pairs[-10:]
            conv = []
            for u, a in last:
                conv.append({"role": "user", "content": u[:500]})
                conv.append({"role": "assistant", "content": a[:500]})
            from memory.mem0_service import mem0_service
            mem0_service.add(conv, user_id=user_id,
                             metadata={"source_session_id": session_id, "auto_extracted": True})
            logger.info(f"[BatchExtract] session={session_id}, {len(last)}轮 → Mem0")
        except Exception as e:
            logger.warning(f"[BatchExtract] 失败: {e}")

    async def delete_session_memory(self, session_id: str, user_id: str = "default_user"):
        if not session_id:
            return
        try:
            ck = await self._checkpointer.aget_tuple({"configurable": {"thread_id": session_id}})
            if ck and ck.checkpoint:
                ms = ck.checkpoint.get("channel_values", {}).get("messages", [])
                if ms:
                    cv = []
                    for m in ms:
                        role = "user" if (hasattr(m, "type") and m.type == "human") else "assistant"
                        if hasattr(m, "content") and m.content:
                            cv.append({"role": role, "content": m.content})
                    if cv:
                        from memory.mem0_service import mem0_service
                        mem0_service.add(cv, user_id=user_id, metadata={"source_session_id": session_id})
                        from memory.memory_cleanup import cleanup
                        cleanup(user_id)
        except Exception as e:
            logger.warning(f"[delete] failed: {e}")
        if hasattr(self._checkpointer, 'delete_thread'):
            self._checkpointer.delete_thread(session_id)
        elif hasattr(self._checkpointer, 'adelete_thread'):
            await self._checkpointer.adelete_thread(session_id)


supervisor_agent = SupervisorAgent()

if __name__ == '__main__':
    async def test():
        print("简单查询:"); await supervisor_agent.run_agent("你好", "t1")
        print("\n" + "=" * 50 + "\n日期:")
        await supervisor_agent.run_agent("今天是几号？", "t2")
    asyncio.run(test())