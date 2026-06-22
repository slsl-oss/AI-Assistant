"""研究报告质量指标：章节覆盖、来源引用、事实准确性、Reviewer 接受率"""

from dataclasses import dataclass, field


@dataclass
class ResearchMetricResults:
    """研究报告评估聚合结果"""
    factual_accuracy: float = 0.0
    section_coverage: float = 0.0
    source_attribution_score: float = 0.0
    reviewer_acceptance_rate: float = 0.0
    avg_revision_count: float = 0.0
    per_topic: list[dict] = field(default_factory=list)
    total_topics: int = 0

    def to_dict(self) -> dict:
        return {
            "factual_accuracy": self.factual_accuracy,
            "section_coverage": self.section_coverage,
            "source_attribution_score": self.source_attribution_score,
            "reviewer_acceptance_rate": self.reviewer_acceptance_rate,
            "avg_revision_count": self.avg_revision_count,
            "total_topics": self.total_topics,
            "per_topic": self.per_topic,
        }


class ResearchMetrics:
    """研究报告指标计算器"""

    @staticmethod
    def compute_section_coverage(
        planned_sections: list[str],
        actual_sections: list[str],
    ) -> float:
        """计算章节覆盖率"""
        if not planned_sections:
            return 1.0
        planned_lower = [s.lower().strip() for s in planned_sections]
        actual_lower = [s.lower().strip() for s in actual_sections]

        covered = 0
        for planned in planned_lower:
            for actual in actual_lower:
                if planned in actual or actual in planned:
                    covered += 1
                    break

        return round(covered / len(planned_sections), 4)

    @staticmethod
    def compute_source_attribution(
        total_claims: int,
        cited_claims: int,
    ) -> float:
        """计算来源引用率"""
        if total_claims == 0:
            return 1.0
        return round(cited_claims / total_claims, 4)

    @staticmethod
    def compute(
        research_results: list[dict],
        judge_scores: list[dict] = None,
    ) -> ResearchMetricResults:
        """聚合所有研究报告指标"""
        results = ResearchMetricResults()
        n = len(research_results)

        if n == 0:
            return results

        results.total_topics = n
        total_coverage = 0.0
        total_accuracy = 0.0
        total_attribution = 0.0
        total_acceptance = 0.0
        total_revisions = 0.0
        acceptance_count = 0

        for i, item in enumerate(research_results):
            coverage = ResearchMetrics.compute_section_coverage(
                item.get("planned_sections", []),
                item.get("actual_sections", []),
            )
            attribution = ResearchMetrics.compute_source_attribution(
                item.get("total_claims", 0),
                item.get("cited_claims", 0),
            )

            total_coverage += coverage
            total_attribution += attribution

            acceptance_rate = item.get("reviewer_acceptance_rate", 0.0)
            if acceptance_rate > 0:
                total_acceptance += acceptance_rate
                acceptance_count += 1

            total_revisions += item.get("avg_revision_count", 0.0)

            judge_score = 0.0
            if judge_scores and i < len(judge_scores):
                judge_score = judge_scores[i].get("accuracy", 0.0)
            total_accuracy += judge_score

            results.per_topic.append({
                "topic": item.get("topic", "")[:80],
                "section_coverage": coverage,
                "source_attribution": attribution,
                "factual_accuracy": judge_score,
                "reviewer_acceptance": item.get("reviewer_acceptance_rate", 0.0),
                "avg_revision_count": item.get("avg_revision_count", 0.0),
            })

        results.section_coverage = round(total_coverage / n, 4)
        results.source_attribution_score = round(total_attribution / n, 4)
        results.factual_accuracy = round(total_accuracy / n, 4)
        results.avg_revision_count = round(total_revisions / n, 2)

        if acceptance_count > 0:
            results.reviewer_acceptance_rate = round(total_acceptance / acceptance_count, 4)

        return results


compute_research_metrics = ResearchMetrics.compute