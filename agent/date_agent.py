import asyncio

from langchain.agents import create_agent
from model.factory import chat_model
from utils.prompts_loader import load_date_prompt
from .tools.middleware import tool_monitor, log_before_model
from .tools.date_agent_tools import date_tools
from langchain.tools import tool


"""
日期查询助手DateAgent
"""
class DateAgent(object):

    def __init__(self):
      self.agent = create_agent(
          model = chat_model,
          system_prompt= load_date_prompt(),
          tools= date_tools,
          middleware= [tool_monitor,log_before_model],
      )



    async def execute_stream(self,query: str):
        input_dict = {
            "messages": [
                {"role": "user", "content": query},
            ]
        }


        # 第三个参数context就是上下文信息，这里我们传递一个字典，包含一个switch_prompt键，值为False,默认使用主提示词
        async for chunk in self.agent.astream(input_dict, stream_mode="values", context={"switch_prompt": False}):
            latest_message = chunk["messages"][-1]
            if latest_message.content:
                # content 可能是列表或字符串，统一处理为字符串
                content = latest_message.content
                if isinstance(content, list):
                    content = "".join(str(c) for c in content)
                yield content.strip()

    async def run_agent(self,query: str):
        chunk_list = []
        async for chunk in self.execute_stream(query):
            print(chunk,end="",flush=True)
            chunk_list.append(chunk)
        return chunk_list[-1]


date_agent = DateAgent()
@tool(
    "date_agent",
    description="日期查询助手,专门用于查询日期相关的信息"
)
async def call_date_agent(query: str):
    return await date_agent.run_agent(query)

if __name__ == '__main__':
    asyncio.run(DateAgent().run_agent("今天日期是多少，星期几"))
