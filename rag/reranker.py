"""
Reranker 精排服务：基于 Cross-Encoder 模型对 RRF 粗排候选进行细粒度重排序，
并通过组合动态阈值过滤低质结果。

两阶段检索流水线：
  Stage 1（粗排）: RRF fusion → 高召回候选集
  Stage 2（精排）: CrossEncoder → 精准排序 + 组合动态阈值过滤 → Top-K

组合动态阈值:
  threshold = max(max_score * score_ratio, abs_threshold)
  - score_ratio: 基于最高分的相对比例
  - abs_threshold: 绝对最低分保底
"""

from typing import List, Optional
from langchain_core.documents import Document
from utils.logger_handler import logger


class RerankerService:
    """单例模式，懒加载 BGE Cross-Encoder 模型"""

    _instance: Optional["RerankerService"] = None

    def __init__(self, model_name: str):
        self._model_name = model_name
        self._model = None

    @classmethod
    def get_instance(cls, model_name: str = "BAAI/bge-reranker-v2-m3") -> "RerankerService":
        """获取单例，若模型名变更则重建"""
        if cls._instance is not None and cls._instance._model_name != model_name:
            cls._instance = None
        if cls._instance is None:
            cls._instance = cls(model_name)
            logger.info(f"[Reranker] 初始化，model: {model_name}")
        return cls._instance

    def _load_model(self):
        """首次调用时加载 CrossEncoder 模型"""
        if self._model is not None:
            return
        try:
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder(
                self._model_name,
                max_length=512,
            )
            logger.info(f"[Reranker] 模型加载完成: {self._model_name}")
        except Exception as e:
            logger.error(f"[Reranker] 模型加载失败: {e}")
            raise

    def rerank(
        self,
        query: str,
        documents: List[Document],
        top_k: int = 3,
        score_ratio: float = 0.3,
        abs_threshold: float = 0.1,
    ) -> List[Document]:
        """
        对候选文档精排 + 组合动态阈值过滤 + Top-K

        Args:
            query: 用户查询文本
            documents: 粗排候选文档列表
            top_k: 最终返回的最大文档数
            score_ratio: 动态阈值 — 基于最高分的相对比例 (0~1)
            abs_threshold: 动态阈值 — 绝对最低分保底 (0~1)

        Returns:
            精排后的文档列表，长度 ≤ top_k，兜底至少 1 条

        阈值公式:
            threshold = max(max_score * score_ratio, abs_threshold)
            取"最高分比例"和"绝对保底"两者中的较大值
        """
        if not documents:
            return []

        if len(documents) == 1:
            return documents

        try:
            self._load_model()

            # 构造 (query, document) 对，截断过长文本
            pairs = [
                (query, doc.page_content[:512])
                for doc in documents
            ]

            # CrossEncoder 批量预测相关性分数
            scores = self._model.predict(
                pairs,
                batch_size=8,
                show_progress_bar=False,
            )

            # 按分数降序排序
            scored = sorted(
                zip(documents, scores),
                key=lambda x: x[1],
                reverse=True,
            )

            max_score = scored[0][1] if scored else 0.0

            # 组合动态阈值
            threshold = max(max_score * score_ratio, abs_threshold)

            # 过滤 + Top-K
            filtered = [
                doc for doc, score in scored
                if score >= threshold
            ]

            if filtered:
                result = filtered[:top_k]
            else:
                # 兜底：全部被过滤时至少返回最高分 1 条
                result = [scored[0][0]]

            logger.info(
                f"[Rerank] {len(documents)} candidates → {len(result)} docs "
                f"(threshold={threshold:.4f}, max_score={max_score:.4f}, "
                f"top_k={top_k})"
            )
            return result

        except Exception as e:
            logger.warning(f"[Rerank] 精排失败，回退原始 RRF 排序: {e}")
            # 降级：返回原始 RRF 排序结果的前 top_k
            return documents[:top_k] if len(documents) > top_k else documents
