"""测试数据集构建器：加载和校验 JSON 格式的测试数据集"""

import json
import os
from dataclasses import dataclass, field
from typing import Optional

from utils.logger_handler import logger


@dataclass
class RAGQueryExample:
    """RAG 检索测试用例"""
    query: str
    relevant_doc_ids: list[str]
    category: str = ""


@dataclass
class AgentTaskExample:
    """Agent 工具选择测试用例"""
    task: str
    expected_tool: str
    expected_tool_category: str = ""
    difficulty: str = "medium"


@dataclass
class ResearchTopicExample:
    """研究报告质量测试用例"""
    topic: str
    expected_sections: list[str]
    expected_min_sources: int = 3
    category: str = ""


class DatasetBuilder:
    """测试数据集加载与校验"""

    @staticmethod
    def load_rag_dataset(path: str) -> list[RAGQueryExample]:
        """加载 RAG 测试查询"""
        if not os.path.exists(path):
            logger.warning(f"[DatasetBuilder] RAG 数据集不存在: {path}")
            return []

        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)

        queries = raw if isinstance(raw, list) else raw.get("rag_queries", [])
        examples = []
        for item in queries:
            examples.append(RAGQueryExample(
                query=item["query"],
                relevant_doc_ids=item.get("relevant_doc_ids", []),
                category=item.get("category", ""),
            ))

        errors = DatasetBuilder.validate_rag_dataset(examples)
        if errors:
            logger.warning(f"[DatasetBuilder] RAG 数据集校验问题: {errors}")

        logger.info(f"[DatasetBuilder] 加载 RAG 数据集: {len(examples)} 条")
        return examples

    @staticmethod
    def load_agent_dataset(path: str) -> list[AgentTaskExample]:
        """加载 Agent 测试任务"""
        if not os.path.exists(path):
            logger.warning(f"[DatasetBuilder] Agent 数据集不存在: {path}")
            return []

        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)

        tasks = raw if isinstance(raw, list) else raw.get("agent_tasks", [])
        examples = []
        for item in tasks:
            examples.append(AgentTaskExample(
                task=item["task"],
                expected_tool=item.get("expected_tool", ""),
                expected_tool_category=item.get("expected_tool_category", ""),
                difficulty=item.get("difficulty", "medium"),
            ))

        logger.info(f"[DatasetBuilder] 加载 Agent 数据集: {len(examples)} 条")
        return examples

    @staticmethod
    def load_research_dataset(path: str) -> list[ResearchTopicExample]:
        """加载研究报告测试主题"""
        if not os.path.exists(path):
            logger.warning(f"[DatasetBuilder] 研究数据集不存在: {path}")
            return []

        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)

        topics = raw if isinstance(raw, list) else raw.get("research_topics", [])
        examples = []
        for item in topics:
            examples.append(ResearchTopicExample(
                topic=item["topic"],
                expected_sections=item.get("expected_sections", []),
                expected_min_sources=item.get("expected_min_sources", 3),
                category=item.get("category", ""),
            ))

        logger.info(f"[DatasetBuilder] 加载研究数据集: {len(examples)} 条")
        return examples

    @staticmethod
    def validate_rag_dataset(examples: list[RAGQueryExample]) -> list[str]:
        """校验 RAG 数据集，返回错误列表"""
        errors = []
        for i, ex in enumerate(examples):
            if not ex.query:
                errors.append(f"第 {i} 条: query 为空")
            if not ex.relevant_doc_ids:
                errors.append(f"第 {i} 条 ({ex.query[:30]}...): relevant_doc_ids 为空")
        return errors