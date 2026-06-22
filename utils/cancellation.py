"""
合作式取消信号机制。

解决两个核心问题：
1. CancelledError 跨层传播：asyncio.CancelledError 会穿透异步生成器到达 LangGraph
   内部，导致 Checkpoint 状态损坏。本模块用合作式检查代替强制异常注入。
2. 信号覆盖：异步生成器被 CancelledError 标记后无法再 yield，导致无法通知
   前端取消状态。本模块用 is_cancelled 属性让生成器在 yield 检查点主动退出。

使用模式：
    token = CancellationToken()
    # 在生成器中
    async for chunk in process():
        if token.is_cancelled:
            yield CANCELLED_SIGNAL
            return
        yield chunk
    # 在取消侧
    token.cancel()  # 同步设置，立即生效
"""

import asyncio


class TaskCancelledError(Exception):
    """合作式取消异常。

    刻意不继承 asyncio.CancelledError：
    - asyncio.CancelledError 会被事件循环特殊处理，在 await 点随机注入
    - 本异常仅在显式检查点抛出，不干扰 LangGraph 内部执行
    """
    pass


class CancellationToken:
    """合作式取消令牌。

    使用 asyncio.Event 实现跨协程信号通知：
    - 执行协程在 yield 检查点查询 is_cancelled
    - 取消协程调用 cancel() 设置信号
    - Event 的 set() 是线程安全的，无需 await

    相比 asyncio.Task.cancel() 的优势：
    - 不注入 CancelledError，保护 LangGraph 状态完整性
    - 异步生成器可以在退出前 yield 取消通知
    - 取消是瞬时的（set 立即生效），响应延迟取决于检查点密度
    """

    def __init__(self):
        self._event = asyncio.Event()
        self._is_cancelled = False

    def cancel(self) -> None:
        """设置取消信号。

        同步方法，可安全地从任何协程或回调中调用。
        set() 后所有等待 wait() 的协程立即被唤醒。
        """
        self._is_cancelled = True
        self._event.set()

    @property
    def is_cancelled(self) -> bool:
        """检查是否已取消。

        供生成器在 yield 检查点使用，不阻塞，不抛异常。
        示例：
            if token.is_cancelled:
                yield CANCELLED_SIGNAL
                return
        """
        return self._is_cancelled

    def throw_if_cancelled(self) -> None:
        """如果已取消则抛出 TaskCancelledError。

        用于需要中断长操作的场景（如 LLM 调用前检查）。
        注意：不应在 LangGraph 执行期间调用，仅在检查点使用。
        """
        if self._is_cancelled:
            raise TaskCancelledError("Task was cancelled by user")

    async def wait(self) -> None:
        """阻塞等待取消信号。

        如果尚未取消，await 会阻塞直到 cancel() 被调用。
        如果已取消，立即返回并抛出 TaskCancelledError。
        """
        await self._event.wait()
        raise TaskCancelledError("Task was cancelled by user")