from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode
from typing import TypedDict, Annotated, List, Literal, Union
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage
import operator
import asyncio
from model.factory import chat_model
from utils.prompts_loader import load_react_prompt, load_date_prompt, load_weather_prompt
from utils.logger_handler import logger
from agent.tools.base_agent_tools import base_tools
from agent.tools.date_agent_tools import date_tools
from agent.tools.weather_agent_tools import weather_tools
from utils.prompts_loader import load_supervisor_prompt


class MultiAgentState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    current_agent: str
    task_results: dict
    final_answer: str
    session_id: str
    user_id: str


def create_agent_node(agent_name: str, system_prompt: str, tools: list):
    """
    创建一个 agent 节点函数

    Args:
        agent_name: agent 名称
        system_prompt: 系统提示词
        tools: 工具列表

    Returns:
        节点函数
    """

    async def agent_node(state: MultiAgentState) -> dict:
        messages = state["messages"]

        if not messages:
            return {"messages": [], "current_agent": agent_name}

        system_message = SystemMessage(content=system_prompt)
        full_messages = [system_message] + messages

        try:
            if tools:
                model_with_tools = chat_model.bind_tools(tools)
                response = await model_with_tools.ainvoke(full_messages)
            else:
                response = await chat_model.ainvoke(full_messages)

            logger.info(f"[{agent_name}] 模型响应: {response.content[:100] if response.content else 'No content'}")

            return {
                "messages": [response],
                "current_agent": agent_name
            }
        except Exception as e:
            logger.error(f"[{agent_name}] 执行失败: {str(e)}")
            error_message = AIMessage(content=f"Agent {agent_name} 执行失败: {str(e)}")
            return {
                "messages": [error_message],
                "current_agent": agent_name
            }

    return agent_node


async def supervisor_node(state: MultiAgentState) -> dict:
    """
    Supervisor 节点：分析用户问题并决定调用哪个 agent
    
    利用 checkpointer 自动恢复的完整对话历史来理解上下文
    """
    messages = state["messages"]

    if not messages:
        return {"current_agent": "react_agent"}

    user_message = None
    history_messages = []
    
    for msg in messages:
        if isinstance(msg, HumanMessage):
            user_message = msg.content
            history_messages.append({"role": "user", "content": msg.content})
        elif isinstance(msg, AIMessage):
            history_messages.append({"role": "assistant", "content": msg.content})

    if not user_message:
        return {"current_agent": "react_agent"}

    system_prompt = load_supervisor_prompt()
    conversation = [{"role": "system", "content": system_prompt}]
    conversation.extend(history_messages)

    try:
        response = await chat_model.ainvoke(conversation)
        decision = response.content.strip().lower()

        valid_agents = ["date_agent", "weather_agent", "react_agent"]
        if decision not in valid_agents:
            decision = "react_agent"

        logger.info(f"[Supervisor] 路由决策: {decision}")

        return {"current_agent": decision}
    except Exception as e:
        logger.error(f"[Supervisor] 路由失败: {str(e)}")
        return {"current_agent": "react_agent"}


def should_continue(state: MultiAgentState) -> Literal["tools", "end"]:
    """
    判断是否需要继续执行工具调用
    """
    messages = state["messages"]
    if not messages:
        return "end"

    last_message = messages[-1]

    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"

    return "end"


def route_to_agent(state: MultiAgentState) -> Literal["date_agent", "weather_agent", "react_agent"]:
    """
    根据 supervisor 的决策路由到相应的 agent
    """
    current_agent = state.get("current_agent", "react_agent")
    return current_agent


class SupervisorAgent:
    """
    基于 LangGraph 的多智能体调度系统
    """

    def __init__(self):
        self._checkpointer = self._init_checkpointer()
        self._graph = self._build_graph()

    def _init_checkpointer(self):
        """初始化 checkpointer"""
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
        """构建多智能体状态图"""

        workflow = StateGraph(MultiAgentState)

        date_agent_node = create_agent_node(
            "date_agent",
            load_date_prompt(),
            date_tools
        )

        weather_agent_node = create_agent_node(
            "weather_agent",
            load_weather_prompt(),
            weather_tools
        )

        react_agent_node = create_agent_node(
            "react_agent",
            load_react_prompt(),
            base_tools
        )

        date_tool_node = ToolNode(date_tools)
        weather_tool_node = ToolNode(weather_tools)
        react_tool_node = ToolNode(base_tools)

        workflow.add_node("supervisor", supervisor_node)
        workflow.add_node("date_agent", date_agent_node)
        workflow.add_node("weather_agent", weather_agent_node)
        workflow.add_node("react_agent", react_agent_node)
        workflow.add_node("date_tools", date_tool_node)
        workflow.add_node("weather_tools", weather_tool_node)
        workflow.add_node("react_tools", react_tool_node)

        workflow.set_entry_point("supervisor")

        # 条件边：Supervisor -> Agent
        workflow.add_conditional_edges(
            "supervisor",
            route_to_agent,
            {
                "date_agent": "date_agent",
                "weather_agent": "weather_agent",
                "react_agent": "react_agent"
            }
        )

        # 条件边：Agent -> Tools 或 END
        workflow.add_conditional_edges(
            "date_agent",
            should_continue,
            {
                "tools": "date_tools",
                "end": END
            }
        )

        workflow.add_conditional_edges(
            "weather_agent",
            should_continue,
            {
                "tools": "weather_tools",
                "end": END
            }
        )

        workflow.add_conditional_edges(
            "react_agent",
            should_continue,
            {
                "tools": "react_tools",
                "end": END
            }
        )

        # 工具执行完成后返回对应 Agent
        workflow.add_edge("date_tools", "date_agent")
        workflow.add_edge("weather_tools", "weather_agent")
        workflow.add_edge("react_tools", "react_agent")

        return workflow.compile(checkpointer=self._checkpointer)

    async def execute_stream(self, query: str, session_id: str = "",
                             user_id: str = "default_user"):
        """
        执行流式查询

        Args:
            query: 用户查询
            session_id: 会话ID
            user_id: 用户ID（用于长期记忆）

        Yields:
            流式输出的文本片段
        """
        from memory.mem0_service import mem0_service

        memories = mem0_service.search(query, user_id=user_id)
        memory_context = mem0_service.format_memories_for_prompt(memories)

        augmented_query = query
        if memory_context:
            augmented_query = f"{memory_context}\n\n[用户当前问题]\n{query}"

        input_state = {
            "messages": [HumanMessage(content=augmented_query)],
            "current_agent": "",
            "task_results": {},
            "final_answer": "",
            "session_id": session_id,
            "user_id": user_id
        }

        config = {
            "configurable": {
                "thread_id": session_id,
                "user_id": user_id
            }
        }
        
        try:
            logger.info(f"[execute_stream] 开始执行查询: {query}")
            
            async for event in self._graph.astream_events(input_state, config, version="v2"):
                kind = event["event"]
                logger.debug(f"[Event] {kind}")
                
                if kind == "on_chat_model_stream":
                    content = event["data"]["chunk"].content
                    if content:
                        logger.debug(f"[Stream] {content}")
                        yield content
                
                elif kind == "on_chat_model_end":
                    logger.debug(f"[Model End] Event data: {event}")
                    
                    output = event.get("data", {}).get("output", None)
                    
                    if not output or not hasattr(output, 'content') or not output.content:
                        continue
                    
                    content = output.content
                    if isinstance(content, list):
                        content = "".join(str(c) for c in content)
                    
                    valid_agents = {"date_agent", "weather_agent", "react_agent"}
                    if content.strip() in valid_agents:
                        logger.info(f"[Model End] 跳过 supervisor 的路由决策: {content}")
                        continue
                    
                    logger.info(f"[Model End] Content: {content}")
                    yield content
                
                elif kind == "on_tool_start":
                    tool_name = event["name"]
                    logger.info(f"[Tool Start] {tool_name}")
                
                elif kind == "on_tool_end":
                    tool_name = event["name"]
                    logger.info(f"[Tool End] {tool_name}")
                
                elif kind == "on_end":
                    logger.info(f"[Graph End] 图执行结束")
            
            logger.info(f"[execute_stream] 执行完成")
        
        except Exception as e:
            logger.error(f"[execute_stream] 执行失败: {str(e)}")
            yield f"执行失败: {str(e)}"

    async def run_agent(self, query: str, session_id: str = "") -> str:
        """
        运行 agent 并返回完整结果

        Args:
            query: 用户查询
            session_id: 会话ID

        Returns:
            完整的响应文本
        """
        result_chunks = []
        async for chunk in self.execute_stream(query, session_id):
            result_chunks.append(chunk)
            print(chunk, end="", flush=True)

        return "".join(result_chunks)

    async def delete_session_memory(self, session_id: str,
                                     user_id: str = "default_user"):
        """
        删除指定会话的记忆，删除前将对话中的关键事实提取到长期记忆

        Args:
            session_id: 会话ID
            user_id: 用户ID（用于长期记忆）
        """
        if not session_id:
            return

        try:
            config = {"configurable": {"thread_id": session_id}}
            checkpoint_tuple = await self._checkpointer.aget_tuple(config)

            if checkpoint_tuple and checkpoint_tuple.checkpoint:
                channel_values = checkpoint_tuple.checkpoint.get("channel_values", {})
                messages = channel_values.get("messages", [])

                if messages:
                    conversation = []
                    for msg in messages:
                        role = "user" if (hasattr(msg, "type") and msg.type == "human") else "assistant"
                        content = msg.content if hasattr(msg, "content") else str(msg)
                        if content:
                            conversation.append({"role": role, "content": content})

                    if conversation:
                        from memory.mem0_service import mem0_service
                        mem0_service.add(
                            conversation,
                            user_id=user_id,
                            metadata={"source_session_id": session_id}
                        )
                        logger.info(
                            f"[delete_session_memory] 已提取长期记忆，"
                            f"session: {session_id}, 消息数: {len(conversation)}"
                        )
        except Exception as e:
            logger.warning(f"[delete_session_memory] 提取长期记忆失败: {e}")

        if hasattr(self._checkpointer, 'delete_thread'):
            self._checkpointer.delete_thread(session_id)
        elif hasattr(self._checkpointer, 'adelete_thread'):
            await self._checkpointer.adelete_thread(session_id)


supervisor_agent = SupervisorAgent()

if __name__ == '__main__':
    async def test():
        print("测试日期查询:")
        await supervisor_agent.run_agent("今天是几号？星期几？", "test_session_1")
        print("\n" + "=" * 50 + "\n")

        print("测试天气查询:")
        await supervisor_agent.run_agent("上海今天天气怎么样？", "test_session_2")
        print("\n" + "=" * 50 + "\n")

        print("测试通用查询:")
        await supervisor_agent.run_agent("你好，介绍一下你自己", "test_session_3")


    asyncio.run(test())
