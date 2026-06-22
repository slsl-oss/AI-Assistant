"""RAG 检索质量评估执行器：直接调用 VectorStoreService 的混合检索"""

import asyncio

from eval.config_loader import EvalConfig
from eval.datasets.builder import RAGQueryExample
from eval.metrics.rag_metrics import RAGMetricResults, compute_rag_metrics
from utils.logger_handler import logger


class RAGRunner:
    """RAG 评估执行器，调用 VectorStoreService 测试检索质量"""

    def __init__(self, config: EvalConfig):
        self.config = config
        self._vector_store = None

    def _init_vector_store(self):
        if self._vector_store is None:
            from rag.vector_stores import VectorStoreService
            self._vector_store = VectorStoreService()
            logger.info("[RAGRunner] VectorStoreService 初始化完成")

    def _extract_doc_ids(self, docs: list) -> list[str]:
        ids = []
        for doc in docs:
            doc_id = ""
            if hasattr(doc, "metadata") and doc.metadata:
                doc_id = doc.metadata.get("source", "") or doc.metadata.get("id", "")
            if not doc_id and hasattr(doc, "id"):
                doc_id = doc.id
            if not doc_id:
                doc_id = str(hash(doc.page_content)) if hasattr(doc, "page_content") else str(id(doc))
            ids.append(doc_id)
        return ids

    async def run(
        self, test_queries: list[RAGQueryExample], compare_reranker: bool = True
    ) -> RAGMetricResults:
        self._init_vector_store()

        if not test_queries:
            logger.warning("[RAGRunner] 无测试查询")
            return RAGMetricResults()

        k_values = self.config.rag.k_values
        candidate_k = self.config.rag.candidate_k
        with_rerank_data = []
        without_rerank_data = []

        for example in test_queries:
            query = example.query
            relevant = example.relevant_doc_ids

            try:
                docs_without = self._vector_store.hybrid_search(query, k=candidate_k)
                retrieved_ids = self._extract_doc_ids(docs_without)
            except Exception as e:
                logger.error(f"[RAGRunner] 检索失败({query[:30]}): {e}")
                retrieved_ids = []

            without_rerank_data.append({
                "query": query, "retrieved_ids": retrieved_ids, "relevant_ids": relevant,
            })

            if compare_reranker:
                try:
                    docs_with = self._vector_store.hybrid_search_with_rerank(query, k=candidate_k)
                    retrieved_ids_rr = self._extract_doc_ids(docs_with)
                except Exception as e:
                    logger.error(f"[RAGRunner] Rerank失败({query[:30]}): {e}")
                    retrieved_ids_rr = []

                with_rerank_data.append({
                    "query": query, "retrieved_ids": retrieved_ids_rr, "relevant_ids": relevant,
                })

        results = compute_rag_metrics(without_rerank_data, k_values)

        if compare_reranker and with_rerank_data:
            results.with_reranker = compute_rag_metrics(with_rerank_data, k_values)
            results.without_reranker = compute_rag_metrics(without_rerank_data, k_values)

        logger.info(
            f"[RAGRunner] 完成: MRR={results.mrr:.4f}, HitRate@3={results.hit_rate.get(3, 0):.4f}"
        )
        return results