import asyncio

from langchain.agents import create_agent
from model.factory import chat_model
from .tools.weather_agent_tools import weather_tools
from utils.prompts_loader import load_weather_prompt
from .tools.middleware import tool_monitor, log_before_model
from langchain.tools import tool


"""
天气查询助手WeatherAgent
"""
class WeatherAgent(object):

    def __init__(self):
      self.agent = create_agent(
          model = chat_model,
          system_prompt= load_weather_prompt(),
          tools= weather_tools,
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



weather_agent = WeatherAgent()
@tool(
    "weather_agent",
    description="天气查询助手,专门用于查询天气相关的信息"
)
async def call_weather_agent(query: str):
    return await weather_agent.run_agent(query)

if __name__ == '__main__':
    asyncio.run(weather_agent.run_agent("今天上海天气怎么样"))
