"""
上下文压缩器：Embedding 相似度过滤，避免上下文爆炸
参考 STORM 架构的两层决策设计
"""
import numpy as np
from langchain_text_splitters import RecursiveCharacterTextSplitter
from model.factory import embedding_model
from utils.logger_handler import logger


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """计算两个向量的余弦相似度"""
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))


class ContextCompressor:
    """上下文压缩器：chunk → embed → similarity filter → top-k"""

    def __init__(
        self,
        similarity_threshold: float = 0.35,
        chunk_size: int = 1000,
        chunk_overlap: int = 100,
        max_results: int = 5,
    ):
        self.threshold = similarity_threshold
        self.max_results = max_results
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", "。", "！", "？", "；", ".", "!", "?", ";", " "],
        )
        self.embeddings = embedding_model

    async def compress(self, query: str, documents: list[dict]) -> dict:
        """
        压缩文档列表，只保留和 query 最相关的段落。

        Returns:
            {text: 压缩后的精华文本, kept: 保留chunk数, total: 总chunk数, sources: 来源URL列表}
        """
        if not documents:
            return {"text": "", "kept": 0, "total": 0, "sources": []}

        total_chars = sum(len(d.get("raw_content", "")) for d in documents)

        # ┌─ 快速通道：总内容 < 8000 字符 → 直接返回原文 ─┐
        if total_chars < 8000 and len(documents) <= self.max_results:
            parts = []
            for d in documents:
                src = d.get("url", "")
                content = d.get("raw_content", "")
                parts.append(f"[来源: {src}]\n{content}")
            text = "\n\n---\n\n".join(parts)
            logger.info(f"[ContextCompressor] 快速通道: {total_chars} chars, {len(documents)} docs")
            return {
                "text": text,
                "kept": len(documents),
                "total": len(documents),
                "sources": [d.get("url", "") for d in documents],
            }

        # ┌─ 标准管道：chunk → embed → filter ─┐
        all_chunks = []
        for doc in documents:
            content = doc.get("raw_content", "")
            if not content:
                continue
            chunks = self.splitter.split_text(content)
            for chunk in chunks:
                all_chunks.append({
                    "content": chunk,
                    "source": doc.get("url", ""),
                    "title": doc.get("title", ""),
                })

        if not all_chunks:
            return {"text": "", "kept": 0, "total": 0, "sources": []}

        # Step 2: 嵌入 query 和所有 chunks
        try:
            query_embedding = np.array(await self.embeddings.aembed_query(query))
            chunk_texts = [c["content"] for c in all_chunks]
            chunk_embeddings = await self.embeddings.aembed_documents(chunk_texts)
        except Exception as e:
            logger.error(f"[ContextCompressor] 嵌入失败: {e}")
            # 降级：直接返回前 max_results 个文档的原文
            parts = []
            for d in documents[:self.max_results]:
                parts.append(f"[来源: {d.get('url', '')}]\n{d.get('raw_content', '')[:2000]}")
            return {
                "text": "\n\n---\n\n".join(parts),
                "kept": min(self.max_results, len(documents)),
                "total": len(documents),
                "sources": [d.get("url", "") for d in documents[:self.max_results]],
            }

        # Step 3: 计算相似度，过滤
        scored = []
        for i, chunk_emb in enumerate(chunk_embeddings):
            sim = _cosine_similarity(query_embedding, np.array(chunk_emb))
            if sim >= self.threshold:
                scored.append((sim, all_chunks[i]))

        # Step 4: 排序，取 top-k
        scored.sort(key=lambda x: x[0], reverse=True)
        top = scored[:self.max_results]

        # Step 5: 拼接
        parts = []
        sources = []
        seen_sources = set()
        for sim, chunk in top:
            src = chunk["source"]
            parts.append(f"[来源: {src} | 相关度: {sim:.2f}]\n{chunk['content']}")
            if src not in seen_sources:
                sources.append(src)
                seen_sources.add(src)

        text = "\n\n---\n\n".join(parts)
        logger.info(
            f"[ContextCompressor] 标准管道: {len(all_chunks)} chunks → "
            f"{len(top)} kept (threshold={self.threshold})"
        )
        return {
            "text": text,
            "kept": len(top),
            "total": len(all_chunks),
            "sources": sources,
        }