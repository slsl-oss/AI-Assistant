"""Agent 回答质量指标：工具选择准确率 + LLM Judge 评分聚合"""

from dataclasses import dataclass, field


@dataclass
class AgentMetricResults:
    """Agent 评估聚合结果"""
    tool_selection_accuracy: float = 0.0
    avg_relevance: float = 0.0
    avg_completeness: float = 0.0
    avg_accuracy: float = 0.0
    avg_safety: float = 0.0
    task_completion_rate: float = 0.0
    overall_score: float = 0.0
    per_task: list[dict] = field(default_factory=list)
    total_tasks: int = 0
    correct_tool_selections: int = 0

    def to_dict(self) -> dict:
        return {
            "tool_selection_accuracy": self.tool_selection_accuracy,
            "avg_relevance": self.avg_relevance,
            "avg_completeness": self.avg_completeness,
            "avg_accuracy": self.avg_accuracy,
            "avg_safety": self.avg_safety,
            "task_completion_rate": self.task_completion_rate,
            "overall_score": self.overall_score,
            "total_tasks": self.total_tasks,
            "correct_tool_selections": self.correct_tool_selections,
            "per_task": self.per_task,
        }


class AgentMetrics:
    """Agent 指标计算器"""

    @staticmethod
    def compute(
        judge_results: list[dict],
        criteria_weights: dict = None,
    ) -> AgentMetricResults:
        """从 LLM Judge 评分结果聚合 Agent 指标"""
        if criteria_weights is None:
            criteria_weights = {
                "relevance": 0.30,
                "completeness": 0.20,
                "accuracy": 0.20,
                "safety": 0.10,
                "tool_selection": 0.20,
            }

        results = AgentMetricResults()
        n = len(judge_results)

        if n == 0:
            return results

        results.total_tasks = n
        total_rel = 0.0
        total_comp = 0.0
        total_acc = 0.0
        total_safe = 0.0
        correct_tools = 0

        for item in judge_results:
            total_rel += item.get("relevance", 0.0)
            total_comp += item.get("completeness", 0.0)
            total_acc += item.get("accuracy", 0.0)
            total_safe += item.get("safety", 0.0)

            if item.get("tool_selection_correct", False):
                correct_tools += 1

            results.per_task.append({
                "task": item.get("task", "")[:80],
                "expected_tool": item.get("expected_tool", ""),
                "actual_tool": item.get("actual_tool", ""),
                "tool_selection_correct": item.get("tool_selection_correct", False),
                "relevance": item.get("relevance", 0.0),
                "completeness": item.get("completeness", 0.0),
                "accuracy": item.get("accuracy", 0.0),
                "safety": item.get("safety", 0.0),
            })

        results.correct_tool_selections = correct_tools
        results.tool_selection_accuracy = round(correct_tools / n, 4)
        results.avg_relevance = round(total_rel / n, 2)
        results.avg_completeness = round(total_comp / n, 2)
        results.avg_accuracy = round(total_acc / n, 2)
        results.avg_safety = round(total_safe / n, 2)
        results.task_completion_rate = round(
            sum(1 for t in judge_results if t.get("completed", False)) / n, 4
        )

        # 加权综合评分
        results.overall_score = round(
            criteria_weights.get("tool_selection", 0.20) * results.tool_selection_accuracy * 5.0 +
            criteria_weights.get("relevance", 0.30) * results.avg_relevance +
            criteria_weights.get("completeness", 0.20) * results.avg_completeness +
            criteria_weights.get("accuracy", 0.20) * results.avg_accuracy +
            criteria_weights.get("safety", 0.10) * results.avg_safety,
            2,
        )

        return results


compute_agent_metrics = AgentMetrics.compute