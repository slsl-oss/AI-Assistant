from typing import Callable

from utils.prompts_loader import load_system_prompt, load_rag_prompt
from langchain.agents import AgentState
from langchain.agents.middleware import wrap_tool_call, before_model, dynamic_prompt, ModelRequest
from langchain.tools.tool_node import ToolCallRequest
from langchain_core.messages import ToolMessage
from langgraph.runtime import Runtime
from langgraph.types import Command
from utils.logger_handler import logger

#同步中间间
@wrap_tool_call  #这个是同步中间件，只有在agent使用invoke 和 stream 同步方法时起效
def tool_monitor(
        # 请求的数据封装
        request: ToolCallRequest,
        # 执行函数本身
        handler: Callable[[ToolCallRequest], ToolMessage | Command],
) -> ToolMessage | Command:   #工具执行的监控
    logger.info(f"[tool monitor]执行工具：{request.tool_call['name']}")
    logger.info(f"[tool monitor]传入参数：{request.tool_call['args']}")

    try:
        result =  handler(request)

        logger.info(f"[tool monitor]调用工具{request.tool_call['name']}成功，")

        if request.tool_call['name'] == "fill_context_for_other_prompt":
            request.runtime.context["switch_prompt"] = True
        return result
    except Exception as e:
        logger.error(f"[tool monitor]调用工具{request.tool_call['name']}失败，原因：{str(e)}")
        raise e

#异步中间件
@wrap_tool_call  #这个是异步中间件，只有在agent使用ainvoke 和 astream 异异步方法时起效
async def tool_monitor(
        # 请求的数据封装
        request: ToolCallRequest,
        # 执行函数本身
        handler: Callable[[ToolCallRequest], ToolMessage | Command],
) -> ToolMessage | Command:   #工具执行的监控
    logger.info(f"[tool monitor]执行工具：{request.tool_call['name']}")
    logger.info(f"[tool monitor]传入参数：{request.tool_call['args']}")

    try:
        result =  await handler(request)

        logger.info(f"[tool monitor]调用工具{request.tool_call['name']}成功，")

        if request.tool_call['name'] == "fill_context_for_other_prompt":
            request.runtime.context["switch_prompt"] = True
        return result
    except Exception as e:
        logger.error(f"[tool monitor]调用工具{request.tool_call['name']}失败，原因：{str(e)}")
        raise e

@before_model
def log_before_model(
        state: AgentState,       #整个agent的状态记录
        runtime: Runtime,        #记录了整个执行过程的上下文信息
):   # 在模型执行前输出日志
    logger.info(f"[log_before_model]即将调用模型，带有{len(state['messages'])}条消息")

    latest_message = state["messages"][-1]
    if latest_message.content:
        # content 可能是列表或字符串，统一处理为字符串
        content = latest_message.content
        if isinstance(content, list):
            content = "".join(str(c) for c in content)
    logger.debug(f"[log_before_model] {type(state['messages'][-1]).__name__}|{content.strip()}")

    return None

@dynamic_prompt     # 每一次生成提示词之前调用此函数
def prompt_switch(request: ModelRequest):   # 动态切换提示词
    is_switch_prompt = request.runtime.context.get("switch_prompt", False)

    if is_switch_prompt:  #为True时，切换提示词
        return load_rag_prompt()

    else:
        return load_system_prompt()


