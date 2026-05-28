import asyncio

from langchain.agents import create_agent
from model.factory import chat_model
from utils.prompts_loader import load_system_prompt
from .tools.base_agent_tools import base_tools
from .tools.middleware import tool_monitor, log_before_model
from .date_agent import call_date_agent
from .weather_agent import call_weather_agent
from langgraph.checkpoint.memory import MemorySaver
from utils.config_handler import agent_conf


"""
langchain框架下的agent 就是基于react架构的agent
流程： 用户传入原始问题 -> 模型根据问题调用工具，获取解决问题所需要的结果 -> 
循环前两步，获取解决问题所需要的全部信息 -> 模型根据全部信息和用户的原始问题进行一个总结融合，通过rag_summarize工具服务检索向量库,生成简要答案
-> 返回给模型 -> 模型根据简要答案，回答用户问题

在本项目中，还有一个rag_summarize工具，用于根据用户的问题，
从数据库中检索出相关的文档，然后进行回答（可以在系统提示词中说明生成最后回答时根据rag_summarize工具生成的简要总结答案回答）
"""
class ReactAgent(object):

    def __init__(self):
        # # 初始化时创建 checkpointer，确保所有请求共享同一个状态存储
        # self._checkpointer = self._init_checkpointer()
        # 初始化时创建 agent，确保所有请求共享同一个 agent 实例
        self._agent = self._create_agent()

    # def _init_checkpointer(self):
    #     """初始化 checkpointer"""
    #     try:
    #         from utils.postgres_checkpointer import AsyncPostgresSaver
    #         db_uri = agent_conf.get("DB_URI")
    #         if db_uri:
    #             saver = AsyncPostgresSaver.from_conn_string(db_uri)
    #             saver.setup()
    #             return saver
    #     except Exception as e:
    #         # 如果PostgreSQL不可用，回退到内存存储
    #         print(f"PostgreSQL连接失败，使用内存存储: {e}")
    #     return MemorySaver()

    def _create_agent(self):
        """创建 agent 实例"""
        return create_agent(
            model=chat_model,
            system_prompt=load_system_prompt(),
            tools=base_tools + [call_date_agent, call_weather_agent],
            middleware=[tool_monitor, log_before_model],
            # checkpointer=self._checkpointer
        )

    async def execute_stream(self, query: str, session_id: str):
        """执行流式查询"""
        # 构建输入消息
        input_dict = {
            "messages": [
                {"role": "user", "content": query},
            ]
        }

        # 构建配置（使用 session_id 作为 thread_id）
        config = {
            "configurable": {
                "thread_id": session_id,
            }
        }

        # stream_mode="values" 返回完整消息列表，取最后一条
        async for chunk in self._agent.astream(input_dict, config, stream_mode="values", context={"switch_prompt": False}):
            latest_message = chunk["messages"][-1]

            # 跳过用户消息和工具消息，只流式输出 AI 回复
            msg_type = getattr(latest_message, "type", "")
            if msg_type in ("human", "tool"):
                continue

            if latest_message.content:
                content = latest_message.content
                if isinstance(content, list):
                    content = "".join(str(c) for c in content)
                text = content.strip()
                if text:
                    yield text

    async def run_agent(self, query: str, session_id: str):
        """运行 agent 并返回完整结果"""
        chunk_list = []
        async for chunk in self.execute_stream(query, session_id):
            print(chunk, end="", flush=True)
            chunk_list.append(chunk)
        return "".join(chunk_list)

    # async def delete_session_memory(self, session_id: str):
    #     """
    #     删除指定会话的记忆（checkpointer中的历史记录）
    #
    #     Args:
    #         session_id: 会话ID（对应 checkpointer 中的 thread_id）
    #     """
    #     if not session_id:
    #         return
    #
    #     # 调用 checkpointer 的 delete_thread 方法删除会话记忆
    #     # thread_id 就是 session_id
    #     if hasattr(self._checkpointer, 'delete_thread'):
    #         self._checkpointer.delete_thread(session_id)
    #     elif hasattr(self._checkpointer, 'adelete_thread'):
    #         await self._checkpointer.adelete_thread(session_id)


if __name__ == '__main__':
    asyncio.run(ReactAgent().run_agent("我的身高175cm，尺码推荐", "test_session_id"))
