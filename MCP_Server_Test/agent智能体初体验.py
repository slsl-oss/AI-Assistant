import asyncio

from langchain.agents import create_agent
from langchain_community.chat_models.tongyi import ChatTongyi
from test_mcp_connection import tools


agent = create_agent(
    model=ChatTongyi(model="qwen3-max"),      #智能体的大脑
    tools=tools,                                  #智能体的工具
    system_prompt="你是一个聊天助手,可以回答用户的问题"
)


async def main():
    res = await agent.ainvoke(
        {
            "messages": [
                {"role": "user", "content": "明天深圳天气怎么样"}
            ]
        }
    )

    for msg in res["messages"]:
        print(type(msg).__name__, msg.content)


if __name__ == '__main__':
    asyncio.run(main())