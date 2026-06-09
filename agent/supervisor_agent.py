import json
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Send
from typing import TypedDict, Annotated, List
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage, ToolMessage
import operator
import asyncio
from model.factory import chat_model
from utils.prompts_loader import load_react_prompt, load_date_prompt, load_weather_prompt, load_supervisor_prompt
from utils.logger_handler import logger
from agent.tools.base_agent_tools import base_tools
from agent.tools.date_agent_tools import date_tools, get_current_time_by_timezone, get_timezone_list, get_city_time, compare_time
from agent.tools.weather_agent_tools import weather_tools as weather_tool_list


class MultiAgentState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    sub_tasks: list
    agent_results: Annotated[list, operator.add]
    final_answer: str
    session_id: str
    user_id: str


DECOMPOSE_PROMPT = load_supervisor_prompt()

AGENT_TOOLS = {
    "date_agent":    {t.name: t for t in date_tools},
    "weather_agent": {t.name: t for t in weather_tool_list},
    "react_agent":   {t.name: t for t in base_tools},
}

AGENT_PROMPTS = {
    "date_agent":    load_date_prompt,
    "weather_agent": load_weather_prompt,
    "react_agent":   load_react_prompt,
}

AGENT_TOOL_LIST = {
    "date_agent":    date_tools,
    "weather_agent": weather_tool_list,
    "react_agent":   base_tools,
}


async def _exec_tool(tool_call: dict, agent_name: str) -> str:
    name = tool_call.get("name", "")
    args = tool_call.get("args", {})
    fn = AGENT_TOOLS.get(agent_name, {}).get(name)
    if not fn:
        return f"未知工具: {name}"
    try:
        if hasattr(fn, 'ainvoke'):
            return str(await asyncio.wait_for(fn.ainvoke(args), timeout=60))
        elif hasattr(fn, 'invoke'):
            loop = asyncio.get_running_loop()
            return str(await loop.run_in_executor(None, fn.invoke, args))
        else:
            return str(fn(**args))
    except asyncio.TimeoutError:
        return f"工具执行超时: {name}"
    except Exception as e:
        return f"工具执行失败: {e}"


async def decompose_query(query: str) -> list[dict]:
    try:
        resp = await chat_model.ainvoke([
            {"role": "system", "content": DECOMPOSE_PROMPT},
            {"role": "user", "content": query}
        ])
        text = resp.content.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1].rsplit("\n", 1)[0]
            if text.startswith("json"):
                text = text[4:]
        tasks = json.loads(text)
        if isinstance(tasks, list) and len(tasks) > 0:
            valid = {"date_agent", "weather_agent", "react_agent"}
            tasks = [t for t in tasks if t.get("agent") in valid]
            if tasks:
                logger.info(f"[Decompose] {len(tasks)} tasks: {[t['agent'] for t in tasks]}")
                return tasks
    except Exception as e:
        logger.warning(f"[Decompose] failed: {e}")
    return [{"agent": "react_agent", "task": query}]


async def supervisor_node(state: MultiAgentState):
    msgs = state["messages"]
    if not msgs:
        return {"sub_tasks": [{"agent": "react_agent", "task": ""}]}
    user_query = ""
    for m in reversed(msgs):
        if isinstance(m, HumanMessage):
            user_query = m.content
            break
    if not user_query:
        return {"sub_tasks": [{"agent": "react_agent", "task": ""}]}
    tasks = await decompose_query(user_query)
    logger.info(f"[Supervisor] {len(tasks)} tasks: {[t['agent'] for t in tasks]}")
    return {"sub_tasks": tasks}


def route_after_supervisor(state: MultiAgentState):
    tasks = state.get("sub_tasks", [])
    if not tasks:
        return "react_agent"
    if len(tasks) == 1:
        return tasks[0]["agent"]
    sends = [Send(t["agent"], {"task": t["task"], "agent_name": t["agent"]}) for t in tasks]
    logger.info(f"[Route] fan-out to {len(sends)} agents")
    return sends


async def agent_node(state: MultiAgentState, agent_name: str):
    task_text = state.get("task", "")
    actual = state.get("agent_name", agent_name)
    prompt_fn = AGENT_PROMPTS.get(actual, load_react_prompt)
    tools_list = AGENT_TOOL_LIST.get(actual, base_tools)

    local = list(state.get("local_messages", []) or [])
    if task_text and not local:
        local = [HumanMessage(content=f"请帮我完成以下任务：{task_text}。直接执行，不要反问或追问。不要使用emoji表情符号。")]
    elif not task_text and not local:
        local = list(state.get("messages", []))

    msgs = [SystemMessage(content=prompt_fn())] + local
    model = chat_model.bind_tools(tools_list) if tools_list else chat_model

    for _ in range(5):
        resp = await model.ainvoke(msgs)
        msgs.append(resp)
        if not (hasattr(resp, "tool_calls") and resp.tool_calls):
            break
        for tc in resp.tool_calls:
            result = await _exec_tool(tc, actual)
            msgs.append(ToolMessage(content=result, tool_call_id=tc["id"]))
            logger.info(f"[{actual}] tool {tc['name']} result: {result[:200]}")

    content = resp.content or ""
    logger.info(f"[{actual}] result: {content[:80]}")

    out = {"agent_results": [{"agent": actual, "task": task_text, "content": content}]}
    if task_text:
        out["local_messages"] = msgs
    else:
        out["messages"] = [resp]
    return out


async def date_agent_node(state):    return await agent_node(state, "date_agent")
async def weather_agent_node(state): return await agent_node(state, "weather_agent")
async def react_agent_node(state):   return await agent_node(state, "react_agent")


async def summarizer_node(state: MultiAgentState) -> dict:
    results = state.get("agent_results", [])
    if not results:
        lm = state.get("messages", [])[-1] if state.get("messages") else None
        c = lm.content if lm and hasattr(lm, "content") else ""
        return {"final_answer": c, "messages": [AIMessage(content=c)] if c else []}
    if len(results) == 1:
        c = results[0].get("content", "")
        resp = await chat_model.ainvoke(f"整理以下回答使其简洁清晰，不要使用emoji表情符号：\n{c}")
        c2 = resp.content or c
        return {"final_answer": c2, "messages": [AIMessage(content=c2)]}
    parts = "\n".join([f"[{r['agent']}]: {r.get('content','')}" for r in results])
    prompt = f"基于以下子任务的回答，合成一个完整连贯的回复给用户。简洁明了，不使用emoji表情符号，也不要提及子任务或agent名称：\n{parts}"
    try:
        resp = await chat_model.ainvoke(prompt)
        c = resp.content or ""
        logger.info(f"[Summarizer] {c[:100]}")
        return {"final_answer": c, "messages": [AIMessage(content=c)]}
    except Exception as e:
        logger.error(f"[Summarizer] failed: {e}")
        return {"final_answer": parts, "messages": [AIMessage(content=parts)]}


class SupervisorAgent:
    def __init__(self):
        self._checkpointer = self._init_checkpointer()
        self._graph = self._build_graph()

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
        wf = StateGraph(MultiAgentState)
        wf.add_node("supervisor", supervisor_node)
        wf.add_node("date_agent", date_agent_node)
        wf.add_node("weather_agent", weather_agent_node)
        wf.add_node("react_agent", react_agent_node)
        wf.add_node("summarizer", summarizer_node)
        wf.set_entry_point("supervisor")
        wf.add_conditional_edges("supervisor", route_after_supervisor,
            {"date_agent": "date_agent", "weather_agent": "weather_agent", "react_agent": "react_agent"})
        for a in ["date_agent", "weather_agent", "react_agent"]:
            wf.add_edge(a, "summarizer")
        wf.add_edge("summarizer", END)
        return wf.compile(checkpointer=self._checkpointer)

    async def execute_stream(self, query: str, session_id: str = "", user_id: str = "default_user"):
        from memory.mem0_service import mem0_service
        mems = mem0_service.search(query, user_id=user_id)
        ctx = mem0_service.format_memories_for_prompt(mems)
        aq = f"{ctx}\n\n[用户当前问题]\n{query}" if ctx else query

        await self._ensure_context_budget(session_id, user_id)

        st = {"messages": [HumanMessage(content=aq)], "sub_tasks": [], "agent_results": [],
              "final_answer": "", "session_id": session_id, "user_id": user_id}
        cfg = {"configurable": {"thread_id": session_id, "user_id": user_id}}
        try:
            async for ev in self._graph.astream_events(st, cfg, version="v2"):
                k = ev["event"]
                if k == "on_chat_model_stream":
                    c = ev["data"]["chunk"].content
                    if c: yield c
                elif k == "on_chat_model_end":
                    o = ev.get("data", {}).get("output", None)
                    if not o or not hasattr(o, 'content') or not o.content: continue
                    c = o.content
                    if isinstance(c, list): c = "".join(str(x) for x in c)
                    if c.strip().startswith("[") and "\"agent\"" in c: continue
                    if not c.strip(): continue
                    node = ev.get("metadata", {}).get("langgraph_node", "")
                    if node in ("date_agent", "weather_agent", "react_agent"):
                        continue
                    yield c
            await self._maybe_batch_extract(session_id, user_id)
        except Exception as e:
            yield f"执行失败: {str(e)}"

    async def _maybe_batch_extract(self, session_id: str, user_id: str):
        try:
            ck = await self._checkpointer.aget_tuple(
                {"configurable": {"thread_id": session_id, "user_id": user_id}})
            if not ck or not ck.checkpoint: return
            msgs = ck.checkpoint.get("channel_values", {}).get("messages", [])
            pairs = []; um = None
            for m in msgs:
                if hasattr(m, "type") and m.type == "human" and m.content: um = m.content
                elif hasattr(m, "type") and m.type == "ai" and m.content and um:
                    pairs.append((um, m.content)); um = None
            if len(pairs) < 10: return
            last = pairs[-10:]; conv = []
            for u, a in last:
                conv.append({"role": "user", "content": u[:500]})
                conv.append({"role": "assistant", "content": a[:500]})
            from memory.mem0_service import mem0_service
            mem0_service.add(conv, user_id=user_id,
                           metadata={"source_session_id": session_id, "auto_extracted": True})
            logger.info(f"[BatchExtract] session={session_id}, {len(last)}轮 → Mem0")
        except Exception as e:
            logger.warning(f"[BatchExtract] 失败: {e}")

    async def _ensure_context_budget(self, session_id: str, user_id: str):
        TOKEN_BUDGET = 20000
        config = {"configurable": {"thread_id": session_id, "user_id": user_id}}
        ck = await self._checkpointer.aget_tuple(config)
        if not ck or not ck.checkpoint: return
        msgs = ck.checkpoint.get("channel_values", {}).get("messages", [])
        if not msgs: return
        pairs = []; um = None
        for m in msgs:
            if hasattr(m, "type") and m.type == "human" and m.content: um = m.content
            elif hasattr(m, "type") and m.type == "ai" and m.content and um:
                pairs.append((um, m.content)); um = None

        def _est(t):
            zh = sum(1 for c in t if '一' <= c <= '鿿')
            return int(zh * 1.5 + (len(t) - zh) * 0.25)

        total = 0; cutoff = 0
        for i in range(len(pairs) - 1, -1, -1):
            u, a = pairs[i]; total += _est(u[:500]) + _est(a[:500])
            if total > TOKEN_BUDGET: cutoff = i + 1; break
        if cutoff == 0: return

        old_summary = ""
        if hasattr(msgs[0], "type") and msgs[0].type == "system":
            old_summary = msgs[0].content
        text = (f"[前序摘要]\n{old_summary}\n\n" if old_summary else "")
        for u, a in pairs[:cutoff]:
            text += f"用户：{u[:500]}\nAI：{a[:500]}\n"
        try:
            resp = await chat_model.ainvoke(
                "总结以下对话的关键信息，保留用户偏好、个人事实、重要决策。用简洁中文。\n\n" + text)
            new_summary = resp.content.strip()
            if not new_summary: return
        except Exception: return
        from langchain_core.messages import SystemMessage
        ck.checkpoint["channel_values"]["messages"] = [
            SystemMessage(content=f"[历史对话摘要]\n{new_summary}")]
        await self._checkpointer.aput(config, ck.checkpoint, ck.metadata,
                                      ck.checkpoint.get("channel_versions", {}))
        logger.info(f"[Compress] session={session_id}, {cutoff}/{len(pairs)}轮→摘要({len(new_summary)}字)")

    async def run_agent(self, query: str, session_id: str = "") -> str:
        r = []
        async for c in self.execute_stream(query, session_id):
            r.append(c); print(c, end="", flush=True)
        return "".join(r)

    async def delete_session_memory(self, session_id: str, user_id: str = "default_user"):
        if not session_id: return
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
        print("简单日期:"); await supervisor_agent.run_agent("今天是几号？星期几？", "t1")
        print("\n" + "=" * 50 + "\n简单天气:")
        await supervisor_agent.run_agent("上海今天天气怎么样？", "t2")
        print("\n" + "=" * 50 + "\n复合:")
        await supervisor_agent.run_agent("今天星期几，上海天气怎么样", "t3")
        print("\n" + "=" * 50 + "\n通用:")
        await supervisor_agent.run_agent("你好", "t4")
    asyncio.run(test())
