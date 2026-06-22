"""端到端性能基准测试执行器：延迟、Token、成本、成功率"""

import asyncio
import time
import uuid

from eval.config_loader import EvalConfig
from eval.metrics.e2e_metrics import E2EMetricResults, compute_e2e_metrics
from utils.logger_handler import logger


class E2ERunner:
    """端到端性能基准测试执行器"""

    def __init__(self, config: EvalConfig):
        self.config = config

    async def run(
        self, query_types: list[str] = None, iterations: int = None, concurrent: int = None
    ) -> E2EMetricResults:
        query_types = query_types or self.config.e2e.query_types
        iterations = iterations or self.config.e2e.iterations_per_type
        concurrent = concurrent or self.config.e2e.concurrent_sessions
        warmup = self.config.e2e.warmup_iterations

        semaphore = asyncio.Semaphore(concurrent)
        all_traces = []

        for qtype in query_types:
            queries = self._get_queries(qtype)
            if not queries:
                continue

            logger.info(f"[E2ERunner] {qtype}: {iterations}次, 并发={concurrent}")

            if warmup > 0:
                warmup_tasks = [
                    self._run_single(queries[0], qtype, semaphore) for _ in range(warmup)
                ]
                await asyncio.gather(*warmup_tasks)

            run_tasks = [
                self._run_single(queries[i % len(queries)], qtype, semaphore)
                for i in range(iterations)
            ]
            traces = await asyncio.gather(*run_tasks)
            all_traces.extend(traces)

        results = compute_e2e_metrics(all_traces)

        logger.info(
            f"[E2ERunner] 完成: 运行={results.total_runs}, 成功率={results.success_rate:.2%}, "
            f"P50={results.overall_latency.p50_ms:.0f}ms, 成本≈¥{results.total_cost_estimate:.4f}"
        )
        return results

    def _get_queries(self, query_type: str) -> list[str]:
        if query_type == "simple":
            return self.config.e2e.simple_queries or ["你好"]
        elif query_type == "deep_research":
            return self.config.e2e.deep_research_queries or ["人工智能发展趋势"]
        return ["你好"]

    async def _run_single(
        self, query: str, query_type: str, semaphore: asyncio.Semaphore
    ) -> dict:
        from agent.supervisor_agent import supervisor_agent

        session_id = f"eval_e2e_{uuid.uuid4().hex[:8]}"
        start = time.time()
        success = True
        error_msg = ""

        async with semaphore:
            try:
                full_response = []
                async for chunk in supervisor_agent.execute_stream(
                    query, session_id, "eval_user"
                ):
                    full_response.append(chunk)
            except Exception as e:
                success = False
                error_msg = str(e)[:200]
                logger.warning(f"[E2ERunner] 失败: {query[:50]} - {e}")

        duration_ms = (time.time() - start) * 1000

        input_tokens = 0
        output_tokens = 0
        try:
            from utils.observability import get_current_trace
            trace = get_current_trace()
            if trace:
                input_tokens = trace.estimated_input_tokens
                output_tokens = trace.estimated_output_tokens
        except Exception:
            pass

        return {
            "trace_id": f"e2e_{session_id}",
            "query": query,
            "query_type": query_type,
            "total_duration_ms": duration_ms,
            "success": success,
            "error": error_msg,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
        }