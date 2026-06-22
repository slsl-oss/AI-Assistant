from typing import Callable
import time

from utils.prompts_loader import load_system_prompt, load_rag_prompt
from langchain.agents import AgentState
from langchain.agents.middleware import wrap_tool_call, before_model, dynamic_prompt, ModelRequest
from langchain.tools.tool_node import ToolCallRequest
from langchain_core.messages import ToolMessage
from langgraph.runtime import Runtime
from langgraph.types import Command
from utils.logger_handler import logger


def _record_tool_trace(name: str, args: dict, success: bool, duration_ms: float, error: str = ""):
    """将工具调用记录到当前追踪上下文"""
    try:
        from utils.observability import get_current_trace
        trace = get_current_trace()
        if trace:
            trace.add_tool_call(name, str(args), success, duration_ms, error)
    except Exception:
        pass


#同步中间件
@wrap_tool_call
def tool_monitor(
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command],
) -> ToolMessage | Command:
    name = request.tool_call.get("name", "unknown")
    args = request.tool_call.get("args", {})
    logger.info(f"[tool monitor]执行工具：{name}")
    logger.info(f"[tool monitor]传入参数：{args}")

    start = time.time()
    try:
        result = handler(request)
        elapsed = (time.time() - start) * 1000
        logger.info(f"[tool monitor]调用工具{name}成功，耗时{elapsed:.0f}ms")
        _record_tool_trace(name, args, True, elapsed)

        if name == "fill_context_for_other_prompt":
            request.runtime.context["switch_prompt"] = True
        return result
    except Exception as e:
        elapsed = (time.time() - start) * 1000
        logger.error(f"[tool monitor]调用工具{name}失败({elapsed:.0f}ms)，原因：{str(e)}")
        _record_tool_trace(name, args, False, elapsed, str(e))
        raise e


#异步中间件
@wrap_tool_call
async def tool_monitor(
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command],
) -> ToolMessage | Command:
    name = request.tool_call.get("name", "unknown")
    args = request.tool_call.get("args", {})
    logger.info(f"[tool monitor]执行工具：{name}")
    logger.info(f"[tool monitor]传入参数：{args}")

    start = time.time()
    try:
        result = await handler(request)
        elapsed = (time.time() - start) * 1000
        logger.info(f"[tool monitor]调用工具{name}成功，耗时{elapsed:.0f}ms")
        _record_tool_trace(name, args, True, elapsed)

        if name == "fill_context_for_other_prompt":
            request.runtime.context["switch_prompt"] = True
        return result
    except Exception as e:
        elapsed = (time.time() - start) * 1000
        logger.error(f"[tool monitor]调用工具{name}失败({elapsed:.0f}ms)，原因：{str(e)}")
        _record_tool_trace(name, args, False, elapsed, str(e))
        raise e

@before_model
def log_before_model(
        state: AgentState,       #整个agent的状态记录
        runtime: Runtime,        #记录了整个执行过程的上下文信息
):   # 在模型执行前输出日志
    logger.info(f"[log_before_model]即将调用模型，带有{len(state['messages'])}条消息")

    latest_message = state["messages"][-1]
    content = ""
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


