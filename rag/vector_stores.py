from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever

from utils.config_handler import chroma_conf
from model.factory import embedding_model
from rag.markdown_chunker import split_markdown
import os
from utils.config_handler import get_abs_path
from utils.file_handler import markitdown_loader, listdir_with_allowed_type, get_file_md5
from utils.logger_handler import logger
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from typing import List


class VectorStoreService(object):
    def __init__(self):

       self.vector_store = Chroma(
           collection_name=chroma_conf["collection_name"],
           embedding_function = embedding_model,
           persist_directory = get_abs_path(chroma_conf["persist_directory"]),
       )

       self._bm25_retriever = None
       self._reranker = None  # RerankerService 懒加载

       self.load_document()

    def _init_bm25(self):
        if self._bm25_retriever is not None:
            return
        all_docs = self.vector_store.get(include=["documents"])
        documents = []
        if all_docs and all_docs.get("documents"):
            for i, content in enumerate(all_docs["documents"]):
                doc_id = all_docs.get("ids", [str(i)])[i] if all_docs.get("ids") else str(i)
                metadata = {}
                if all_docs.get("metadatas") and i < len(all_docs["metadatas"]):
                    metadata = all_docs["metadatas"][i] or {}
                documents.append(Document(page_content=content, metadata=metadata, id=doc_id))
        if documents:
            self._bm25_retriever = BM25Retriever.from_documents(documents)
            self._bm25_retriever.k = chroma_conf["k"]
            logger.info(f"[VectorStore] BM25 retriever ready, docs: {len(documents)}")

    def _init_reranker(self):
        """懒加载 RerankerService 单例"""
        if self._reranker is None and chroma_conf.get("rerank_enabled", True):
            from rag.reranker import RerankerService
            self._reranker = RerankerService.get_instance(
                chroma_conf.get("rerank_model", "BAAI/bge-reranker-v2-m3")
            )

    def _build_filter(self, user_id=None):
        if user_id:
            return {"$or": [{"user_id": user_id}, {"user_id": "__shared__"}]}
        return {"user_id": "__shared__"}  # 不传 user_id 只看共享文档

    def get_retriever(self, user_id=None):
        """
        获得单路向量库检索的retriever
        :param user_id:
        :return:
        """
        kwargs = {"k": chroma_conf["k"]}
        f = self._build_filter(user_id)
        if f:
            kwargs["filter"] = f
        return self.vector_store.as_retriever(search_kwargs=kwargs)

    def add_user_document(self, file_path: str, user_id: str):
        from utils.file_handler import markitdown_loader as loader

        filename = os.path.basename(file_path)

        existing = self.vector_store.get(where={"$and": [{"user_id": user_id}, {"source": filename}]})
        if existing and existing.get("ids") and len(existing["ids"]) > 0:
            logger.info(f"[VectorStore] File {filename} already exists for user {user_id}, skip")
            return -1

        docs = loader(file_path)
        if not docs:
            return 0
        split_docs = split_markdown(
            docs,
            chunk_tokens=chroma_conf.get("chunk_tokens", 500),
            overlap_sections=1
        )
        for doc in split_docs:
            doc.metadata["user_id"] = user_id
            doc.metadata["source"] = filename
        self.vector_store.add_documents(split_docs)
        self._bm25_retriever = None
        logger.info(f"[VectorStore] Added {len(split_docs)} chunks from {file_path} for user {user_id}")
        return len(split_docs)

    def get_bm25_retriever(self):
        self._init_bm25()
        return self._bm25_retriever

    def hybrid_search(self, query: str, k: int = None, rrf_k: int = 60) -> List[Document]:
        """
        BM25 + vector retrieval + RRF fusion.

        RRF formula: score(d) = sum(1 / (k + rank_i(d))) for each retriever i
        """
        if k is None:
            k = chroma_conf["k"]

        self._init_bm25()
        vector_retriever = self.vector_store.as_retriever(search_kwargs={"k": k * 2})

        vector_docs = vector_retriever.invoke(query)
        bm25_docs = self._bm25_retriever.invoke(query) if self._bm25_retriever else []

        doc_scores = {}
        for rank, doc in enumerate(vector_docs):
            doc_id = doc.id if hasattr(doc, 'id') and doc.id else doc.page_content[:100]
            doc_scores[doc_id] = {"doc": doc, "score": 0}
            doc_scores[doc_id]["score"] += 1.0 / (rrf_k + rank + 1)

        for rank, doc in enumerate(bm25_docs):
            doc_id = doc.id if hasattr(doc, 'id') and doc.id else doc.page_content[:100]
            if doc_id not in doc_scores:
                doc_scores[doc_id] = {"doc": doc, "score": 0}
            doc_scores[doc_id]["score"] += 1.0 / (rrf_k + rank + 1)

        sorted_items = sorted(doc_scores.values(), key=lambda x: x["score"], reverse=True)
        return [item["doc"] for item in sorted_items[:k]]

    def hybrid_search_with_rerank(self, query: str, k: int = None, rrf_k: int = 60,
                                   user_id: str = None) -> List[Document]:
        """
        两阶段检索：RRF 粗排（高召回） → Reranker 精排（高精度） → 动态阈值过滤

        Args:
            query: 用户查询文本
            k: 最终返回数量（默认取 chroma_conf["k"]）
            rrf_k: RRF 融合参数
            user_id: 用户 ID（用于过滤用户专属文档）

        Returns:
            精排后的文档列表
        """
        if k is None:
            k = chroma_conf["k"]

        coarse_k = k * chroma_conf.get("coarse_k_multiplier", 3)

        # Stage 1: RRF 粗排（取更多候选保证召回率）
        candidates = self.hybrid_search(query, k=coarse_k, rrf_k=rrf_k)

        if not candidates:
            logger.info("[Rerank] 粗排无候选结果")
            return []

        if len(candidates) <= k:
            # 候选数不超过目标数时无需精排
            return candidates

        # Stage 2: Reranker 精排 + 组合动态阈值过滤
        self._init_reranker()

        if self._reranker is None:
            # 未启用精排时回退
            return candidates[:k]

        return self._reranker.rerank(
            query=query,
            documents=candidates,
            top_k=chroma_conf.get("rerank_top_k", k),
            score_ratio=chroma_conf.get("rerank_score_ratio", 0.3),
            abs_threshold=chroma_conf.get("rerank_abs_threshold", 0.1),
        )

    def get_hybrid_retriever(self, user_id: str = None) -> BaseRetriever:
        """
        获得混合检索的 retriever，根据 rerank_enabled 配置自动选择：
        - 启用精排：hybrid_search_with_rerank（RRF 粗排 → Reranker 精排）
        - 未启用：  hybrid_search（纯 RRF 融合）

        :param user_id: 用户 ID（用于过滤用户专属文档）
        :return:
        """
        vs = self
        rerank_enabled = chroma_conf.get("rerank_enabled", True)

        class HybridRetriever(BaseRetriever):
            def _get_relevant_documents(self, query: str) -> List[Document]:
                if rerank_enabled:
                    return vs.hybrid_search_with_rerank(query, user_id=user_id)
                return vs.hybrid_search(query)

        return HybridRetriever()

    def add_report_to_store(self, title: str, content: str, user_id: str = "__shared__"):
        """
        将生成的报告存入向量数据库，供后续 RAG 检索。

        Args:
            title: 报告标题
            content: 报告全文（Markdown）
            user_id: 用户 ID
        """
        if not content or not content.strip():
            return 0

        source = f"[报告] {title}"
        doc = Document(page_content=content, metadata={
            "user_id": user_id, "source": source, "type": "generated_report",
        })
        split_docs = split_markdown(
            [doc],
            chunk_tokens=chroma_conf.get("chunk_tokens", 500),
            overlap_sections=1
        )
        for d in split_docs:
            d.metadata["user_id"] = user_id
            d.metadata["source"] = source
            d.metadata["type"] = "generated_report"

        self.vector_store.add_documents(split_docs)
        self._bm25_retriever = None
        logger.info(f"[VectorStore] 报告已存入: {source} ({len(split_docs)} chunks)")
        return len(split_docs)

    def load_document(self):

        def check_md5(md5_str: str):
            if not os.path.exists(get_abs_path(chroma_conf["md5_hex_store"])):
                open(get_abs_path(chroma_conf["md5_hex_store"]), 'w', encoding="utf-8").close
                return False
            else:
                with open(get_abs_path(chroma_conf["md5_hex_store"]), 'r', encoding="utf-8") as f:
                    lines = f.readlines()
                    for line in lines:
                        if line.strip() == md5_str:
                            return True
                return False

        def save_md5(md5_str: str):
            with open(get_abs_path(chroma_conf["md5_hex_store"]), 'a', encoding="utf-8") as f:
                f.write(md5_str + "\n")

        allowed_files_path = listdir_with_allowed_type(
            get_abs_path(chroma_conf["data_path"]),
            tuple(chroma_conf["allow_knowledge_type"])
        )

        for path in allowed_files_path:
            md5_hex = get_file_md5(path)
            if check_md5(md5_hex):
                logger.info(f"[load doc] file {path} already in store, skip")
                continue
            try:
                documents: list[Document] = markitdown_loader(path)
                if not documents:
                    logger.warning(f"[load doc] file {path} empty, skip")
                    continue
                split_docs: list[Document] = split_markdown(
                    documents,
                    chunk_tokens=chroma_conf.get("chunk_tokens", 500),
                    overlap_sections=1
                )
                if not split_docs:
                    logger.warning(f"[load doc] file {path} empty after split, skip")
                for doc in split_docs:
                    doc.metadata["user_id"] = "__shared__"
                self.vector_store.add_documents(split_docs)
                save_md5(md5_hex)
                logger.info(f"[load doc] file {path} loaded into store ({len(split_docs)} chunks)")
            except Exception as e:
                logger.error(f"[load doc] file {path} load failed: {str(e)}", exc_info=True)
                continue


if __name__ == '__main__':
    vs = VectorStoreService()
    vs.load_document()
    retriever = vs.get_retriever()
    res = retriever.invoke("my weight is 180 jin, size recommendation")
    print(res)
