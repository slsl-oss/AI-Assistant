"""评估框架 CLI 入口

Usage:
    python -m eval.cli rag              # 仅 RAG 评估
    python -m eval.cli agent            # 仅 Agent 评估
    python -m eval.cli research         # 仅研究报告评估
    python -m eval.cli e2e              # 仅端到端基准测试
    python -m eval.cli all              # 全部评估
    python -m eval.cli all --quick      # 快速模式
"""

import argparse
import asyncio
import sys

from eval.config_loader import load_eval_config, EvalConfig
from eval.datasets.builder import DatasetBuilder
from eval.runners.rag_runner import RAGRunner
from eval.runners.agent_runner import AgentRunner
from eval.runners.research_runner import ResearchRunner
from eval.runners.e2e_runner import E2ERunner
from eval.reporter import ReportGenerator
from eval.metrics.rag_metrics import RAGMetricResults
from eval.metrics.agent_metrics import AgentMetricResults
from eval.metrics.research_metrics import ResearchMetricResults
from eval.metrics.e2e_metrics import E2EMetricResults
from utils.logger_handler import logger


async def run_rag_eval(config: EvalConfig) -> RAGMetricResults:
    logger.info("=" * 50)
    logger.info("RAG 检索质量评估")
    dataset = DatasetBuilder.load_rag_dataset(config.datasets["rag_queries"])
    if not dataset:
        logger.warning("RAG 数据集为空，跳过")
        return RAGMetricResults()
    runner = RAGRunner(config)
    results = await runner.run(dataset, compare_reranker=config.rag.compare_reranker)
    logger.info(f"RAG 完成: MRR={results.mrr:.4f}")
    return results


async def run_agent_eval(config: EvalConfig) -> AgentMetricResults:
    logger.info("=" * 50)
    logger.info("Agent 回答质量评估")
    dataset = DatasetBuilder.load_agent_dataset(config.datasets["agent_tasks"])
    if not dataset:
        logger.warning("Agent 数据集为空，跳过")
        return AgentMetricResults()
    runner = AgentRunner(config)
    results = await runner.run(dataset)
    logger.info(f"Agent 完成: 综合={results.overall_score:.2f}")
    return results


async def run_research_eval(config: EvalConfig) -> ResearchMetricResults:
    logger.info("=" * 50)
    logger.info("研究报告质量评估")
    dataset = DatasetBuilder.load_research_dataset(config.datasets["research_topics"])
    if not dataset:
        logger.warning("研究数据集为空，跳过")
        return ResearchMetricResults()
    runner = ResearchRunner(config)
    results = await runner.run(dataset)
    logger.info(f"研究完成: 覆盖率={results.section_coverage:.2%}")
    return results


async def run_e2e_eval(config: EvalConfig, quick: bool = False) -> E2EMetricResults:
    logger.info("=" * 50)
    logger.info("端到端性能基准测试")
    iterations = 2 if quick else config.e2e.iterations_per_type
    runner = E2ERunner(config)
    results = await runner.run(
        query_types=config.e2e.query_types,
        iterations=iterations,
        concurrent=config.e2e.concurrent_sessions,
    )
    logger.info(f"E2E 完成: P50={results.overall_latency.p50_ms:.0f}ms")
    return results


async def run_all(config: EvalConfig, quick: bool = False) -> str:
    logger.info("=" * 60)
    logger.info("AI Assistant 全量评估")
    if quick:
        config.rag.k_values = [1, 3, 5]
        config.e2e.iterations_per_type = 2
        logger.info("快速模式")

    results = {}
    rag_task = asyncio.create_task(run_rag_eval(config))
    e2e_task = asyncio.create_task(run_e2e_eval(config, quick))
    results["rag"] = await rag_task
    results["e2e"] = await e2e_task
    results["agent"] = await run_agent_eval(config)
    results["research"] = await run_research_eval(config)

    reporter = ReportGenerator(config)
    report_path = reporter.generate(
        rag_results=results["rag"],
        agent_results=results["agent"],
        research_results=results["research"],
        e2e_results=results["e2e"],
    )
    logger.info(f"全量评估完成，报告: {report_path}")
    return report_path


def main():
    parser = argparse.ArgumentParser(description="AI Assistant 评估框架")
    parser.add_argument("command", choices=["rag", "agent", "research", "e2e", "all"], help="评估类型")
    parser.add_argument("--config", default=None, help="配置文件路径")
    parser.add_argument("--quick", action="store_true", help="快速模式")
    parser.add_argument("--output", default=None, help="自定义报告名称")
    args = parser.parse_args()

    config = load_eval_config(args.config)
    if args.output:
        config.output.run_id = args.output

    try:
        if args.command == "rag":
            asyncio.run(run_rag_eval(config))
        elif args.command == "agent":
            asyncio.run(run_agent_eval(config))
        elif args.command == "research":
            asyncio.run(run_research_eval(config))
        elif args.command == "e2e":
            asyncio.run(run_e2e_eval(config, args.quick))
        elif args.command == "all":
            asyncio.run(run_all(config, args.quick))
    except KeyboardInterrupt:
        logger.info("评估被用户中断")
        sys.exit(0)
    except Exception as e:
        logger.error(f"评估失败: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()