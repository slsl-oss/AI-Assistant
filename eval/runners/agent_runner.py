"""Agent 回答质量评估执行器：通过 supervisor_agent 执行 + LLM Judge 评分"""

import asyncio
import uuid

from eval.config_loader import EvalConfig
from eval.datasets.builder import AgentTaskExample
from eval.judges.llm_judge import LLMJudge
from eval.metrics.agent_metrics import AgentMetricResults, compute_agent_metrics
from utils.logger_handler import logger


class AgentRunner:
    """Agent 评估执行器，测试工具选择准确率和回答质量"""

    def __init__(self, config: EvalConfig):
        self.config = config
        self.judge = LLMJudge(config)

    async def run(self, test_tasks: list[AgentTaskExample]) -> AgentMetricResults:
        if not test_tasks:
            logger.warning("[AgentRunner] 无测试任务")
            return AgentMetricResults()

        semaphore = asyncio.Semaphore(self.config.agent.max_concurrent_tasks)
        judge_results = []

        async def run_one(task: AgentTaskExample):
            async with semaphore:
                return await self._evaluate_task(task)

        judge_results = await asyncio.gather(*[run_one(t) for t in test_tasks])

        results = compute_agent_metrics(
            list(judge_results), self.config.agent.criteria_weights
        )
        logger.info(
            f"[AgentRunner] 完成: 工具准确率={results.tool_selection_accuracy:.2%}, "
            f"综合={results.overall_score:.2f}"
        )
        return results

    async def _evaluate_task(self, task: AgentTaskExample) -> dict:
        from agent.supervisor_agent import supervisor_agent

        session_id = f"eval_agent_{uuid.uuid4().hex[:8]}"
        actual_tool = ""
        response_text = ""
        completed = False

        try:
            async with asyncio.timeout(self.config.agent.task_timeout_seconds):
                full_response = []
                async for chunk in supervisor_agent.execute_stream(
                    task.task, session_id, "eval_user"
                ):
                    full_response.append(chunk)
                response_text = "".join(full_response)
                completed = True

                from utils.observability import get_current_trace
                trace = get_current_trace()
                if trace and trace.tool_calls:
                    actual_tool = trace.tool_calls[0].name if trace.tool_calls else ""

        except asyncio.TimeoutError:
            response_text = "[超时]"
            logger.warning(f"[AgentRunner] 超时: {task.task[:50]}")
        except Exception as e:
            response_text = f"[错误: {str(e)[:100]}]"
            logger.error(f"[AgentRunner] 失败: {task.task[:50]} - {e}")

        tool_correct = (actual_tool == task.expected_tool) if task.expected_tool else (actual_tool == "")

        judge_scores = {}
        if completed and response_text and response_text not in ("[超时]",):
            try:
                judge_scores = await self.judge.evaluate_response(task.task, response_text[:2000])
            except Exception as e:
                logger.warning(f"[AgentRunner] Judge失败: {e}")

        return {
            "task": task.task,
            "expected_tool": task.expected_tool,
            "actual_tool": actual_tool,
            "tool_selection_correct": tool_correct,
            "completed": completed,
            "relevance": judge_scores.get("relevance", 0.0),
            "completeness": judge_scores.get("completeness", 0.0),
            "accuracy": judge_scores.get("accuracy", 0.0),
            "safety": judge_scores.get("safety", 0.0),
        }