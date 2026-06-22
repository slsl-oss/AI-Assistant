"""
可观测性追踪模块：记录每次请求的完整调用链路

追踪维度：
  - 调用链路：哪个 agent，走的什么路径
  - 模型：用的什么模型
  - 工具调用：名称、参数、成功/失败、耗时
  - 失败节点：发生在哪一步、错误信息
  - 响应时间：总耗时、各阶段耗时
  - Token 估算：输入/输出 token
"""
import contextvars
import time
import uuid
from dataclasses import dataclass, field
from typing import Optional

from utils.logger_handler import logger


@dataclass
class ToolCallRecord:
    name: str
    args: str
    success: bool
    duration_ms: float
    error: str = ""


@dataclass
class StepRecord:
    name: str
    start_time: float
    end_time: float = 0.0
    success: bool = True
    error: str = ""

    @property
    def duration_ms(self) -> float:
        end = self.end_time or time.time()
        return (end - self.start_time) * 1000


@dataclass
class TraceContext:
    """单次请求的完整追踪上下文"""
    trace_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    session_id: str = ""
    user_id: str = ""
    query: str = ""
    start_time: float = field(default_factory=time.time)
    end_time: float = 0.0

    agent_path: str = ""
    model: str = ""
    is_deep_research: bool = False

    steps: list[StepRecord] = field(default_factory=list)
    _current_step: Optional[StepRecord] = None

    tool_calls: list[ToolCallRecord] = field(default_factory=list)
    errors: list[dict] = field(default_factory=list)

    estimated_input_tokens: int = 0
    estimated_output_tokens: int = 0

    @property
    def total_duration_ms(self) -> float:
        end = self.end_time or time.time()
        return (end - self.start_time) * 1000

    @property
    def tool_success_rate(self) -> float:
        if not self.tool_calls:
            return 1.0
        return sum(1 for t in self.tool_calls if t.success) / len(self.tool_calls)

    def start_step(self, name: str):
        self._current_step = StepRecord(name=name, start_time=time.time())
        self.steps.append(self._current_step)

    def end_step(self, success: bool = True, error: str = ""):
        if self._current_step:
            self._current_step.end_time = time.time()
            self._current_step.success = success
            self._current_step.error = error
            self._current_step = None

    def add_tool_call(self, name: str, args: str, success: bool, duration_ms: float, error: str = ""):
        self.tool_calls.append(ToolCallRecord(
            name=name, args=args[:200], success=success, duration_ms=duration_ms, error=error,
        ))

    def add_error(self, step: str, message: str):
        self.errors.append({"step": step, "message": message[:300]})

    def _estimate_tokens(self, text: str) -> int:
        if not text:
            return 0
        zh = sum(1 for c in text if '一' <= c <= '鿿')
        return int(zh * 1.5 + (len(text) - zh) * 0.25)

    def add_input_tokens(self, text: str):
        self.estimated_input_tokens += self._estimate_tokens(text)

    def add_output_tokens(self, text: str):
        self.estimated_output_tokens += self._estimate_tokens(text)

    def finish(self):
        self.end_time = time.time()
        self.end_step()

    def summary(self) -> str:
        lines = [
            "=" * 60,
            f"Trace: {self.trace_id} | Session: {self.session_id}",
            f"Query: {self.query[:80]}",
            f"Path: {self.agent_path}",
            f"Model: {self.model}",
            f"Duration: {self.total_duration_ms:.0f}ms",
            f"Steps: {len(self.steps)} ({sum(1 for s in self.steps if s.success)} ok, {sum(1 for s in self.steps if not s.success)} fail)",
        ]
        for s in self.steps:
            status = "ok" if s.success else "FAIL"
            err = f" ({s.error})" if s.error else ""
            lines.append(f"  [{status}] {s.name}: {s.duration_ms:.0f}ms{err}")

        if self.tool_calls:
            lines.append(f"Tools: {len(self.tool_calls)} ({self.tool_success_rate:.0%} success)")
            for t in self.tool_calls:
                status = "ok" if t.success else "FAIL"
                lines.append(f"  [{status}] {t.name}({t.args[:60]}): {t.duration_ms:.0f}ms")

        if self.errors:
            lines.append(f"Errors: {len(self.errors)}")
            for e in self.errors:
                lines.append(f"  FAIL [{e['step']}] {e['message'][:120]}")

        lines.append(f"Tokens: ~{self.estimated_input_tokens} in / ~{self.estimated_output_tokens} out")
        lines.append("=" * 60)
        return "\n".join(lines)


_current_trace: contextvars.ContextVar[Optional[TraceContext]] = contextvars.ContextVar(
    "current_trace", default=None
)


def get_current_trace() -> Optional[TraceContext]:
    return _current_trace.get()


def set_current_trace(trace: TraceContext):
    _current_trace.set(trace)


def clear_current_trace():
    _current_trace.set(None)


def log_trace_summary(trace: TraceContext):
    logger.info(f"\n{trace.summary()}")