"""
研究流水线：三级并行引擎
  L1: asyncio.gather 子问题并行
  L2: 多搜索引擎并行
  L3: WorkerPool(Semaphore(15)) URL 并行抓取
  → ContextCompressor → LLM 写草稿
"""
import asyncio
import json
import os
import re
from typing import Optional

import requests
from model.factory import research_agent_model
from utils.logger_handler import logger
from agent.infrastructure.web_scraper import WebScraper
from agent.infrastructure.context_compressor import ContextCompressor
from rag.rag_service import RagSummarizeService


class ResearchPipeline:
    """研究流水线：搜索 → 抓取 → 压缩 → LLM 草稿"""

    def __init__(self):
        self.scraper = WebScraper(max_workers=15)
        self.compressor = ContextCompressor()
        self.model = research_agent_model
        self._rag_service: Optional[RagSummarizeService] = None

    def _get_rag_service(self) -> RagSummarizeService:
        if self._rag_service is None:
            self._rag_service = RagSummarizeService()
        return self._rag_service

    async def _get_rag_context(self, query: str) -> str:
        """从本地 RAG 知识库检索相关文档，返回格式化文本。无结果时返回空字符串。"""
        try:
            rag = self._get_rag_service()
            context = await asyncio.to_thread(rag.local_rag_context, query)
            return context or ""
        except Exception as e:
            logger.warning(f"[Pipeline] RAG 检索失败: {e}")
            return ""

    async def close(self):
        await self.scraper.close()

    # ==================== Step 0: 生成子问题 ====================

    async def _generate_sub_queries(self, query: str, max_sub: int = 5) -> list[str]:
        """LLM 根据原始问题生成 3-5 个相关子问题"""
        prompt = (
            "你是一个研究规划助手。根据用户的问题，生成 3-5 个相关的子问题，"
            "每个子问题从不同角度深入探索。\n\n"
            "返回格式：严格返回 JSON 字符串数组，不要包含其他内容。\n\n"
            f"用户问题：{query}\n\n"
            "子问题（JSON数组）："
        )
        try:
            resp = await self.model.ainvoke(prompt)
            text = resp.content.strip()
            if text.startswith("```"):
                text = text.split("\n", 1)[1].rsplit("\n", 1)[0]
                if text.startswith("json"):
                    text = text[4:]
            sub_queries = json.loads(text)
            if isinstance(sub_queries, list) and len(sub_queries) > 0:
                logger.info(f"[Pipeline] 生成 {len(sub_queries)} 个子问题: {sub_queries}")
                return sub_queries[:max_sub]
        except Exception as e:
            logger.warning(f"[Pipeline] 子问题生成失败: {e}")
        return [query]

    # ==================== Step 1: 搜索 ====================

    async def _search(self, query: str) -> list[str]:
        """多搜索引擎并行搜索，返回 URL 列表"""
        urls = []

        # 百度搜索
        try:
            api_key = os.getenv("BAIDU_API_KEY")
            if api_key:
                url = "https://qianfan.baidubce.com/v2/ai_search/web_search"
                headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
                data = {"messages": [{"content": query}]}
                resp = await asyncio.to_thread(
                    requests.post, url, headers=headers, json=data, timeout=30
                )
                if resp.status_code == 200:
                    result = resp.json()
                    extracted = self._extract_urls_from_baidu(result)
                    urls.extend(extracted)
                    logger.info(f"[Pipeline] 百度搜索: {len(extracted)} URLs")
        except Exception as e:
            logger.warning(f"[Pipeline] 百度搜索失败: {e}")

        # 去重
        seen = set()
        unique = []
        for u in urls:
            if u not in seen:
                seen.add(u)
                unique.append(u)
        return unique

    def _extract_urls_from_baidu(self, result: dict) -> list[str]:
        """从百度搜索结果 JSON 中提取 URL"""
        urls = []
        try:
            if "pages" in result:
                for page in result["pages"]:
                    if isinstance(page, dict) and "url" in page:
                        urls.append(page["url"])
            elif "results" in result:
                for r in result["results"]:
                    if isinstance(r, dict) and "url" in r:
                        urls.append(r["url"])
            elif "messages" in result:
                for msg in result["messages"]:
                    if isinstance(msg, dict) and "content" in msg:
                        found = re.findall(r'\[.*?\]\((https?://[^)]+)\)', str(msg["content"]))
                        urls.extend(found)
        except Exception as e:
            logger.warning(f"[Pipeline] URL提取失败: {e}")
        return urls

    # ==================== Step 2+3: 抓取 + 压缩 ====================

    async def _process_sub_query(self, sub_query: str) -> dict:
        """处理单个子问题：搜索 → 抓取 → 压缩 → LLM 草稿"""
        urls = await self._search(sub_query)
        if not urls:
            return {
                "query": sub_query,
                "draft": f"未找到关于 '{sub_query}' 的搜索结果。",
                "sources": [],
                "stats": {"urls_found": 0, "scraped": 0, "chunks_kept": 0},
            }

        # L3: 并行抓取
        docs = await self.scraper.scrape_many(urls[:10])
        if not docs:
            return {
                "query": sub_query,
                "draft": f"抓取 '{sub_query}' 相关网页失败。",
                "sources": urls[:5],
                "stats": {"urls_found": len(urls), "scraped": 0, "chunks_kept": 0},
            }

        # 压缩
        compressed = await self.compressor.compress(sub_query, docs)

        # LLM 写草稿
        if compressed["text"]:
            draft = await self._write_draft(sub_query, compressed["text"])
        else:
            draft = f"关于 '{sub_query}' 未找到足够相关内容。"

        return {
            "query": sub_query,
            "draft": draft,
            "sources": compressed.get("sources", []),
            "stats": {
                "urls_found": len(urls),
                "scraped": len(docs),
                "chunks_kept": compressed.get("kept", 0),
                "total_chunks": compressed.get("total", 0),
            },
        }

    async def _write_draft(self, query: str, context: str) -> str:
        """LLM 根据网页压缩文档 + RAG 知识库文档写草稿"""
        rag_context = await self._get_rag_context(query)

        # 构建研究资料：网页抓取 + RAG 知识库
        research_materials = ""
        if context:
            research_materials += f"## 网页抓取资料\n{context}\n\n"
        if rag_context:
            research_materials += f"## 本地知识库资料\n{rag_context}\n\n"
        if not research_materials:
            research_materials = "无相关研究资料"

        prompt = (
            "你是一个专业的研究助手。根据以下研究资料，围绕问题撰写一份简洁的草稿。\n\n"
            "要求：\n"
            "- 提取关键信息，组织成连贯的段落\n"
            "- 网页资料标注来源 URL，知识库资料标注【参考资料N】\n"
            "- 融合网页和知识库的信息，互补不足\n"
            "- 不确定的信息标注 '待验证'\n"
            "- 不使用 emoji 表情符号\n\n"
            f"## 问题\n{query}\n\n"
            f"{research_materials}"
            "## 草稿"
        )
        try:
            resp = await self.model.ainvoke(prompt)
            return resp.content.strip()
        except Exception as e:
            logger.error(f"[Pipeline] 写草稿失败: {e}")
            return f"生成草稿失败: {str(e)}"

    # ==================== 主入口：三级并行 ====================

    async def process(self, query: str) -> dict:
        """主入口：L1 子问题并行处理"""
        sub_queries = await self._generate_sub_queries(query)

        # L1: asyncio.gather 子问题并行
        logger.info(f"[Pipeline] L1 并行: {len(sub_queries)} 个子问题")
        sub_results = await asyncio.gather(
            *[self._process_sub_query(sq) for sq in sub_queries]
        )

        merged = await self._merge_drafts(query, sub_results)

        return {
            "query": query,
            "sub_results": sub_results,
            "merged_draft": merged,
            "all_sources": list(set(
                s for r in sub_results for s in r.get("sources", [])
            )),
        }

    async def _merge_drafts(self, query: str, sub_results: list[dict]) -> str:
        """合并所有子问题草稿为最终报告，同时融入 RAG 知识库资料"""
        if len(sub_results) == 1:
            return sub_results[0]["draft"]

        parts = []
        for i, r in enumerate(sub_results, 1):
            parts.append(f"### 子问题 {i}: {r['query']}\n{r['draft']}")

        combined = "\n\n".join(parts)

        rag_context = await self._get_rag_context(query)
        rag_section = ""
        if rag_context:
            rag_section = f"## 本地知识库补充资料\n{rag_context}\n\n"

        prompt = (
            "你是一个专业的研究报告撰写助手。将以下多个子问题的研究结果合并为一份连贯的报告。\n\n"
            "要求：\n"
            "- 按逻辑顺序重组内容，避免重复\n"
            "- 保留所有来源引用\n"
            "- 如有本地知识库资料，与网页资料融合互补\n"
            "- 使用 Markdown 格式，层次分明\n"
            "- 不使用 emoji 表情符号\n\n"
            f"## 原始问题\n{query}\n\n"
            f"## 各子问题研究结果\n{combined}\n\n"
            f"{rag_section}"
            "## 合并报告"
        )
        try:
            resp = await self.model.ainvoke(prompt)
            return resp.content.strip()
        except Exception as e:
            logger.error(f"[Pipeline] 合并失败: {e}")
            return combined