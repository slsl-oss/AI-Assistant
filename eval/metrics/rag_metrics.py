"""RAG 检索质量指标：纯函数计算，无副作用"""

import math
from dataclasses import dataclass, field
from typing import Optional


def hit_rate_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    """top-k 中至少命中一个相关文档的比例（单次查询返回 0 或 1）"""
    top_k = retrieved_ids[:k]
    return 1.0 if any(doc_id in relevant_ids for doc_id in top_k) else 0.0


def mrr(retrieved_ids: list[str], relevant_ids: set[str]) -> float:
    """Mean Reciprocal Rank: 1 / 第一个相关文档的排名。无命中返回 0"""
    for i, doc_id in enumerate(retrieved_ids, 1):
        if doc_id in relevant_ids:
            return 1.0 / i
    return 0.0


def ndcg_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    """Normalized Discounted Cumulative Gain@k，使用二元相关性（1/0）"""
    dcg = 0.0
    for i, doc_id in enumerate(retrieved_ids[:k], 1):
        if doc_id in relevant_ids:
            dcg += 1.0 / math.log2(i + 1)

    ideal_count = min(len(relevant_ids), k)
    idcg = 0.0
    for i in range(1, ideal_count + 1):
        idcg += 1.0 / math.log2(i + 1)

    return dcg / idcg if idcg > 0 else 0.0


def precision_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    """top-k 中相关文档的比例"""
    top_k = retrieved_ids[:k]
    if k == 0:
        return 0.0
    return len([d for d in top_k if d in relevant_ids]) / k


def recall_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    """所有相关文档中被检索到的比例"""
    if not relevant_ids:
        return 1.0
    top_k = set(retrieved_ids[:k])
    return len(top_k & relevant_ids) / len(relevant_ids)


@dataclass
class RAGMetricResults:
    """RAG 评估聚合结果"""
    hit_rate: dict[int, float] = field(default_factory=dict)
    mrr: float = 0.0
    ndcg: dict[int, float] = field(default_factory=dict)
    precision: dict[int, float] = field(default_factory=dict)
    recall: dict[int, float] = field(default_factory=dict)
    per_query: list[dict] = field(default_factory=list)
    with_reranker: Optional["RAGMetricResults"] = None
    without_reranker: Optional["RAGMetricResults"] = None

    def to_dict(self) -> dict:
        result = {
            "hit_rate": self.hit_rate,
            "mrr": self.mrr,
            "ndcg": self.ndcg,
            "precision": self.precision,
            "recall": self.recall,
            "per_query": self.per_query,
        }
        if self.with_reranker:
            result["with_reranker"] = self.with_reranker.to_dict()
        if self.without_reranker:
            result["without_reranker"] = self.without_reranker.to_dict()
        return result


class RAGMetrics:
    """RAG 指标计算器"""

    @staticmethod
    def compute(
        queries: list[dict],
        k_values: list[int],
    ) -> RAGMetricResults:
        """对所有查询聚合所有 RAG 指标"""
        results = RAGMetricResults()
        per_query = []

        total_hit = {k: 0.0 for k in k_values}
        total_mrr = 0.0
        total_ndcg = {k: 0.0 for k in k_values}
        total_precision = {k: 0.0 for k in k_values}
        total_recall = {k: 0.0 for k in k_values}
        n = len(queries)

        if n == 0:
            return results

        for q in queries:
            retrieved = q.get("retrieved_ids", [])
            relevant = set(q.get("relevant_ids", []))

            q_result = {"query": q.get("query", "")[:80]}
            for k in k_values:
                h = hit_rate_at_k(retrieved, relevant, k)
                n_score = ndcg_at_k(retrieved, relevant, k)
                p = precision_at_k(retrieved, relevant, k)
                r = recall_at_k(retrieved, relevant, k)

                total_hit[k] += h
                total_ndcg[k] += n_score
                total_precision[k] += p
                total_recall[k] += r

                q_result[f"hit@{k}"] = h
                q_result[f"ndcg@{k}"] = round(n_score, 4)
                q_result[f"precision@{k}"] = round(p, 4)
                q_result[f"recall@{k}"] = round(r, 4)

            m = mrr(retrieved, relevant)
            total_mrr += m
            q_result["mrr"] = round(m, 4)

            per_query.append(q_result)

        for k in k_values:
            results.hit_rate[k] = round(total_hit[k] / n, 4)
            results.ndcg[k] = round(total_ndcg[k] / n, 4)
            results.precision[k] = round(total_precision[k] / n, 4)
            results.recall[k] = round(total_recall[k] / n, 4)

        results.mrr = round(total_mrr / n, 4)
        results.per_query = per_query

        return results


compute_rag_metrics = RAGMetrics.compute