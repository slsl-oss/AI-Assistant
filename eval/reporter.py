"""评估报告生成器：JSON + Markdown 双格式输出"""

import json
import os
from datetime import datetime
from typing import Optional

from eval.config_loader import EvalConfig
from eval.metrics.rag_metrics import RAGMetricResults
from eval.metrics.agent_metrics import AgentMetricResults
from eval.metrics.research_metrics import ResearchMetricResults
from eval.metrics.e2e_metrics import E2EMetricResults
from utils.config_handler import get_abs_path
from utils.logger_handler import logger


class ReportGenerator:
    """评估报告生成器"""

    def __init__(self, config: EvalConfig):
        self.config = config
        self.output_dir = get_abs_path(config.output.reports_dir)
        os.makedirs(self.output_dir, exist_ok=True)

    def generate(
        self,
        rag_results: Optional[RAGMetricResults] = None,
        agent_results: Optional[AgentMetricResults] = None,
        research_results: Optional[ResearchMetricResults] = None,
        e2e_results: Optional[E2EMetricResults] = None,
        run_id: str = "",
    ) -> str:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_id = run_id or timestamp

        json_path = os.path.join(self.output_dir, f"eval_{run_id}.json")
        md_path = os.path.join(self.output_dir, f"eval_{run_id}.md")

        self._write_json(json_path, run_id, rag_results, agent_results, research_results, e2e_results)
        self._write_markdown(md_path, run_id, rag_results, agent_results, research_results, e2e_results)

        logger.info(f"[ReportGenerator] 报告已生成: {self.output_dir}")
        return self.output_dir

    def _write_json(self, path, run_id, rag, agent, research, e2e):
        report = {
            "run_id": run_id,
            "timestamp": datetime.now().isoformat(),
            "rag": rag.to_dict() if rag else None,
            "agent": agent.to_dict() if agent else None,
            "research": research.to_dict() if research else None,
            "e2e": e2e.to_dict() if e2e else None,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2, default=str)

    def _write_markdown(self, path, run_id, rag, agent, research, e2e):
        sections = [
            f"# AI Assistant 评估报告\n\n**运行ID:** {run_id}\n**时间:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n",
            self._md_rag_section(rag),
            self._md_agent_section(agent),
            self._md_research_section(research),
            self._md_e2e_section(e2e),
            self._md_summary(rag, agent, research, e2e),
        ]
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n\n---\n\n".join(s for s in sections if s))

    def _md_rag_section(self, results: Optional[RAGMetricResults]) -> str:
        if not results:
            return ""
        lines = ["## RAG 检索质量\n"]
        k_values = sorted(results.hit_rate.keys()) if results.hit_rate else [1, 3, 5, 10]

        header = "| 指标 | " + " | ".join(f"k={k}" for k in k_values) + " |"
        sep = "|---|" + "|".join("---" for _ in k_values) + "|"
        lines.append(header)
        lines.append(sep)

        lines.append("| Hit Rate | " + " | ".join(f"{results.hit_rate.get(k, 0):.2%}" for k in k_values) + " |")
        lines.append("| NDCG | " + " | ".join(f"{results.ndcg.get(k, 0):.4f}" for k in k_values) + " |")
        lines.append("| Precision | " + " | ".join(f"{results.precision.get(k, 0):.2%}" for k in k_values) + " |")
        lines.append("| Recall | " + " | ".join(f"{results.recall.get(k, 0):.2%}" for k in k_values) + " |")
        lines.append(f"\n**MRR:** {results.mrr:.4f}")

        if results.with_reranker and results.without_reranker:
            lines.append(f"\n### Reranker 对比")
            lines.append(f"| 指标 | 无 Reranker | 有 Reranker | 提升 |")
            lines.append(f"|---|---|---|---|")
            for k in k_values:
                without = results.without_reranker.hit_rate.get(k, 0)
                with_rr = results.with_reranker.hit_rate.get(k, 0)
                diff = with_rr - without
                sign = "+" if diff >= 0 else ""
                lines.append(f"| Hit Rate@{k} | {without:.2%} | {with_rr:.2%} | {sign}{diff:.2%} |")
            without_mrr = results.without_reranker.mrr
            with_mrr = results.with_reranker.mrr
            diff_mrr = with_mrr - without_mrr
            sign = "+" if diff_mrr >= 0 else ""
            lines.append(f"| MRR | {without_mrr:.4f} | {with_mrr:.4f} | {sign}{diff_mrr:.4f} |")

        return "\n".join(lines)

    def _md_agent_section(self, results: Optional[AgentMetricResults]) -> str:
        if not results:
            return ""
        return "\n".join([
            "## Agent 回答质量\n",
            f"| 指标 | 得分 |",
            f"|---|---|",
            f"| 工具选择准确率 | {results.tool_selection_accuracy:.2%} ({results.correct_tool_selections}/{results.total_tasks}) |",
            f"| 平均相关性 | {results.avg_relevance:.2f}/5.0 |",
            f"| 平均完整性 | {results.avg_completeness:.2f}/5.0 |",
            f"| 平均准确性 | {results.avg_accuracy:.2f}/5.0 |",
            f"| 平均安全性 | {results.avg_safety:.2f}/5.0 |",
            f"| 任务完成率 | {results.task_completion_rate:.2%} |",
            f"| **综合评分** | **{results.overall_score:.2f}/5.0** |",
        ])

    def _md_research_section(self, results: Optional[ResearchMetricResults]) -> str:
        if not results:
            return ""
        return "\n".join([
            "## 研究报告质量\n",
            f"| 指标 | 得分 |",
            f"|---|---|",
            f"| 章节覆盖率 | {results.section_coverage:.2%} |",
            f"| 来源引用率 | {results.source_attribution_score:.2%} |",
            f"| 事实准确性 | {results.factual_accuracy:.2%} |",
            f"| Reviewer 接受率 | {results.reviewer_acceptance_rate:.2%} |",
            f"| 平均修订次数 | {results.avg_revision_count:.1f} |",
            f"| 测试主题数 | {results.total_topics} |",
        ])

    def _md_e2e_section(self, results: Optional[E2EMetricResults]) -> str:
        if not results:
            return ""
        lines = [
            "## 端到端性能\n",
            "### 延迟分布\n",
            "| 类型 | P50 | P95 | P99 | Mean | Min | Max |",
            "|---|---|---|---|---|---|---|",
        ]
        for qtype, stats in results.latency_by_query_type.items():
            lines.append(
                f"| {qtype} | {stats.p50_ms:.0f}ms | {stats.p95_ms:.0f}ms | "
                f"{stats.p99_ms:.0f}ms | {stats.mean_ms:.0f}ms | {stats.min_ms:.0f}ms | {stats.max_ms:.0f}ms |"
            )
        overall = results.overall_latency
        lines.append(
            f"| **总计** | **{overall.p50_ms:.0f}ms** | **{overall.p95_ms:.0f}ms** | "
            f"**{overall.p99_ms:.0f}ms** | **{overall.mean_ms:.0f}ms** | **{overall.min_ms:.0f}ms** | **{overall.max_ms:.0f}ms** |"
        )

        lines.append(f"\n### Token 用量与成本\n")
        lines.append(f"| 类型 | 平均输入Token | 平均输出Token | 运行次数 | 成本(¥) |")
        lines.append(f"|---|---|---|---|---|")
        for qtype, usage in results.token_usage.items():
            cost = results.cost_estimate.get(qtype, 0)
            lines.append(
                f"| {qtype} | {usage['avg_input_tokens']:.0f} | "
                f"{usage['avg_output_tokens']:.0f} | {usage['total_runs']} | ¥{cost:.4f} |"
            )
        lines.append(f"\n| **总计** | - | - | {results.total_runs} | **¥{results.total_cost_estimate:.4f}** |")
        lines.append(f"\n### 成功率\n")
        lines.append(f"| 成功率 | 失败率 | 成功/总数 |")
        lines.append(f"|---|---|---|")
        lines.append(f"| {results.success_rate:.2%} | {results.failure_rate:.2%} | {results.success_count}/{results.total_runs} |")
        return "\n".join(lines)

    def _md_summary(self, rag, agent, research, e2e) -> str:
        lines = ["## 总结\n"]
        lines.append("| 评估维度 | 核心指标 | 结果 |")
        lines.append("|---|---|---|")
        if rag:
            lines.append(f"| RAG 检索 | MRR | {rag.mrr:.4f} |")
            h3 = rag.hit_rate.get(3, 0)
            lines.append(f"| RAG 检索 | Hit Rate@3 | {h3:.2%} |")
        if agent:
            lines.append(f"| Agent 质量 | 综合评分 | {agent.overall_score:.2f}/5.0 |")
            lines.append(f"| Agent 质量 | 工具准确率 | {agent.tool_selection_accuracy:.2%} |")
        if research:
            lines.append(f"| 研究报告 | 章节覆盖率 | {research.section_coverage:.2%} |")
            lines.append(f"| 研究报告 | 事实准确性 | {research.factual_accuracy:.2%} |")
        if e2e:
            lines.append(f"| 端到端 | P50延迟 | {e2e.overall_latency.p50_ms:.0f}ms |")
            lines.append(f"| 端到端 | 成功率 | {e2e.success_rate:.2%} |")
            lines.append(f"| 端到端 | 总成本 | ¥{e2e.total_cost_estimate:.4f} |")
        return "\n".join(lines)