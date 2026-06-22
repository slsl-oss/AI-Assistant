"""端到端性能指标：延迟分布、Token 用量、成功率、成本估算"""

import statistics
from dataclasses import dataclass, field


@dataclass
class LatencyStats:
    """延迟统计"""
    p50_ms: float = 0.0
    p95_ms: float = 0.0
    p99_ms: float = 0.0
    mean_ms: float = 0.0
    min_ms: float = 0.0
    max_ms: float = 0.0

    def to_dict(self) -> dict:
        return {
            "p50_ms": self.p50_ms,
            "p95_ms": self.p95_ms,
            "p99_ms": self.p99_ms,
            "mean_ms": self.mean_ms,
            "min_ms": self.min_ms,
            "max_ms": self.max_ms,
        }


@dataclass
class E2EMetricResults:
    """端到端评估聚合结果"""
    latency_by_query_type: dict[str, LatencyStats] = field(default_factory=dict)
    overall_latency: LatencyStats = field(default_factory=LatencyStats)
    token_usage: dict[str, dict] = field(default_factory=dict)
    success_rate: float = 0.0
    failure_rate: float = 0.0
    cost_estimate: dict[str, float] = field(default_factory=dict)
    total_cost_estimate: float = 0.0
    traces: list[dict] = field(default_factory=list)
    total_runs: int = 0
    success_count: int = 0

    def to_dict(self) -> dict:
        return {
            "latency_by_query_type": {
                k: v.to_dict() for k, v in self.latency_by_query_type.items()
            },
            "overall_latency": self.overall_latency.to_dict(),
            "token_usage": self.token_usage,
            "success_rate": self.success_rate,
            "failure_rate": self.failure_rate,
            "cost_estimate": self.cost_estimate,
            "total_cost_estimate": self.total_cost_estimate,
            "total_runs": self.total_runs,
            "success_count": self.success_count,
        }


class E2EMetrics:
    """端到端指标计算器"""

    # DeepSeek-v4 定价（元/百万 tokens）
    INPUT_PRICE_PER_1M = 2.0
    OUTPUT_PRICE_PER_1M = 8.0

    @staticmethod
    def compute_latency_stats(durations_ms: list[float]) -> LatencyStats:
        """计算延迟分布"""
        if not durations_ms:
            return LatencyStats()

        sorted_durations = sorted(durations_ms)
        n = len(sorted_durations)

        return LatencyStats(
            p50_ms=round(sorted_durations[int(n * 0.50)], 0),
            p95_ms=round(sorted_durations[min(int(n * 0.95), n - 1)], 0),
            p99_ms=round(sorted_durations[min(int(n * 0.99), n - 1)], 0),
            mean_ms=round(statistics.mean(sorted_durations), 0),
            min_ms=round(min(sorted_durations), 0),
            max_ms=round(max(sorted_durations), 0),
        )

    @staticmethod
    def estimate_cost(input_tokens: int, output_tokens: int) -> float:
        """估算单次查询成本（元）"""
        input_cost = (input_tokens / 1_000_000) * E2EMetrics.INPUT_PRICE_PER_1M
        output_cost = (output_tokens / 1_000_000) * E2EMetrics.OUTPUT_PRICE_PER_1M
        return round(input_cost + output_cost, 6)

    @staticmethod
    def compute(
        traces: list[dict],
        query_type_map: dict[str, str] = None,
    ) -> E2EMetricResults:
        """从 TraceContext 数据聚合端到端指标"""
        results = E2EMetricResults()

        if not traces:
            return results

        results.total_runs = len(traces)
        all_durations = []
        type_durations: dict[str, list[float]] = {}
        type_tokens: dict[str, dict] = {}
        total_input_tokens = 0
        total_output_tokens = 0

        for trace in traces:
            query_type = trace.get("query_type", "unknown")
            duration = trace.get("total_duration_ms", 0.0)
            success = trace.get("success", True)
            input_tokens = trace.get("input_tokens", 0)
            output_tokens = trace.get("output_tokens", 0)

            all_durations.append(duration)

            if query_type not in type_durations:
                type_durations[query_type] = []
                type_tokens[query_type] = {
                    "total_input": 0, "total_output": 0, "count": 0,
                }

            type_durations[query_type].append(duration)
            type_tokens[query_type]["total_input"] += input_tokens
            type_tokens[query_type]["total_output"] += output_tokens
            type_tokens[query_type]["count"] += 1

            total_input_tokens += input_tokens
            total_output_tokens += output_tokens

            if success:
                results.success_count += 1

            results.traces.append({
                "trace_id": trace.get("trace_id", ""),
                "query": trace.get("query", "")[:80],
                "query_type": query_type,
                "duration_ms": duration,
                "success": success,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
            })

        results.overall_latency = E2EMetrics.compute_latency_stats(all_durations)
        results.success_rate = round(results.success_count / results.total_runs, 4)
        results.failure_rate = round(1.0 - results.success_rate, 4)

        for qtype, durations in type_durations.items():
            results.latency_by_query_type[qtype] = E2EMetrics.compute_latency_stats(durations)

            tokens = type_tokens[qtype]
            count = tokens["count"]
            results.token_usage[qtype] = {
                "avg_input_tokens": round(tokens["total_input"] / count, 0) if count > 0 else 0,
                "avg_output_tokens": round(tokens["total_output"] / count, 0) if count > 0 else 0,
                "total_runs": count,
            }

            results.cost_estimate[qtype] = E2EMetrics.estimate_cost(
                tokens["total_input"], tokens["total_output"],
            )

        results.total_cost_estimate = round(sum(results.cost_estimate.values()), 6)

        return results


compute_e2e_metrics = E2EMetrics.compute