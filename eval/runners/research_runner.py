"""研究报告质量评估执行器：执行 STORM 流水线 + 解析报告 + LLM Judge 评估"""

import asyncio
import re
import uuid

from eval.config_loader import EvalConfig
from eval.datasets.builder import ResearchTopicExample
from eval.judges.llm_judge import LLMJudge
from eval.metrics.research_metrics import ResearchMetricResults, compute_research_metrics
from utils.logger_handler import logger


class ResearchRunner:
    """研究报告评估执行器"""

    def __init__(self, config: EvalConfig):
        self.config = config
        self.judge = LLMJudge(config)

    def _parse_sections(self, report: str) -> list[str]:
        return re.findall(r'^#{1,3}\s+(.+)$', report, re.MULTILINE)

    def _parse_sources(self, report: str) -> list[str]:
        sources = []
        source_matches = re.findall(r'\[(\d+)\]\s*(https?://[^\s\)]+)', report)
        sources = [url for _, url in source_matches]
        ref_section = re.search(r'参考(?:来源|文献|资料).*?\n(.*?)$', report, re.DOTALL)
        if ref_section:
            urls = re.findall(r'https?://[^\s\)]+', ref_section.group(1))
            sources.extend(urls)
        return list(set(sources))

    def _estimate_claims(self, report: str) -> tuple[int, int]:
        sentences = re.split(r'[。！？\n]', report)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
        total = len(sentences)
        cited = sum(1 for s in sentences if re.search(r'\[\d+\]|\([^)]*\d{4}[^)]*\)', s))
        return total, cited

    async def run(self, test_topics: list[ResearchTopicExample]) -> ResearchMetricResults:
        if not test_topics:
            logger.warning("[ResearchRunner] 无测试主题")
            return ResearchMetricResults()

        research_results = []
        judge_scores = []

        for topic in test_topics:
            logger.info(f"[ResearchRunner] 评估: {topic.topic[:50]}...")
            try:
                result = await self._evaluate_topic(topic)
                research_results.append(result)
            except Exception as e:
                logger.error(f"[ResearchRunner] 失败: {topic.topic[:50]} - {e}")
                research_results.append({
                    "topic": topic.topic,
                    "planned_sections": topic.expected_sections,
                    "actual_sections": [],
                    "total_claims": 0,
                    "cited_claims": 0,
                    "reviewer_acceptance_rate": 0.0,
                    "avg_revision_count": 0.0,
                })

        for i, result in enumerate(research_results):
            report_text = result.get("report", "")
            sources = result.get("sources", [])
            if report_text and sources:
                try:
                    judge_result = await self.judge.evaluate_factual_accuracy(
                        report_text[:self.config.research.max_report_chars], sources
                    )
                    judge_scores.append(judge_result)
                except Exception as e:
                    logger.warning(f"[ResearchRunner] Judge失败: {e}")
                    judge_scores.append({"accuracy": 0.5})

        results = compute_research_metrics(research_results, judge_scores)
        logger.info(
            f"[ResearchRunner] 完成: 覆盖率={results.section_coverage:.2%}, "
            f"准确性={results.factual_accuracy:.2%}"
        )
        return results

    async def _evaluate_topic(self, topic: ResearchTopicExample) -> dict:
        from agent.supervisor_agent import supervisor_agent

        session_id = f"eval_research_{uuid.uuid4().hex[:8]}"
        report_text = ""
        actual_sections = []
        sources = []

        try:
            full_response = []
            async for chunk in supervisor_agent.execute_stream(
                topic.topic, session_id, "eval_user"
            ):
                full_response.append(chunk)
            report_text = "".join(full_response)
            actual_sections = self._parse_sections(report_text)
            sources = self._parse_sources(report_text)
        except Exception as e:
            logger.error(f"[ResearchRunner] 执行失败: {e}")
            report_text = f"[执行失败: {str(e)[:200]}]"

        total_claims, cited_claims = self._estimate_claims(report_text)

        reviewer_acceptance = 0.0
        avg_revisions = 0.0
        try:
            from utils.observability import get_current_trace
            trace = get_current_trace()
            if trace:
                review_steps = [s for s in trace.steps if "review" in s.name.lower()]
                if review_steps:
                    accept_count = sum(1 for s in review_steps if s.success)
                    reviewer_acceptance = accept_count / len(review_steps)
                    avg_revisions = len([s for s in trace.steps if "revis" in s.name.lower()])
        except Exception:
            pass

        return {
            "topic": topic.topic,
            "planned_sections": topic.expected_sections,
            "actual_sections": actual_sections,
            "total_claims": total_claims,
            "cited_claims": cited_claims,
            "reviewer_acceptance_rate": reviewer_acceptance,
            "avg_revision_count": avg_revisions,
            "report": report_text,
            "sources": sources,
        }