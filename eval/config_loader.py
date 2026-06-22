"""评估配置加载器，遵循项目 config/*.yaml 模式"""

import os
import yaml
from dataclasses import dataclass, field
from typing import Optional

from utils.config_handler import get_abs_path
from utils.logger_handler import logger


@dataclass
class RAGConfig:
    k_values: list[int] = field(default_factory=lambda: [1, 3, 5, 10])
    compare_reranker: bool = True
    rrf_k: int = 60
    candidate_k: int = 15


@dataclass
class AgentConfig:
    judge_model_role: str = "summarizer"
    max_concurrent_tasks: int = 5
    task_timeout_seconds: int = 120
    evaluation_criteria: list[str] = field(default_factory=lambda: ["relevance", "completeness", "accuracy", "safety"])
    criteria_weights: dict = field(default_factory=lambda: {
        "relevance": 0.30, "completeness": 0.20, "accuracy": 0.20,
        "safety": 0.10, "tool_selection": 0.20,
    })


@dataclass
class ResearchConfig:
    judge_model_role: str = "reviewer"
    min_section_coverage: float = 0.8
    requires_source_attribution: bool = True
    reviewer_acceptance_threshold: float = 0.7
    max_report_chars: int = 50000


@dataclass
class E2EConfig:
    query_types: list[str] = field(default_factory=lambda: ["simple", "deep_research"])
    iterations_per_type: int = 5
    concurrent_sessions: int = 2
    simple_queries: list[str] = field(default_factory=list)
    deep_research_queries: list[str] = field(default_factory=list)
    warmup_iterations: int = 1


@dataclass
class JudgeConfig:
    model_name: str = "deepseek-v4-flash"
    prompts_dir: str = "eval/judges/prompts"


@dataclass
class OutputConfig:
    reports_dir: str = "eval/reports"
    format: str = "both"
    timestamp_suffix: bool = True
    include_per_query_details: bool = True


@dataclass
class EvalConfig:
    datasets: dict = field(default_factory=dict)
    rag: RAGConfig = field(default_factory=RAGConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    research: ResearchConfig = field(default_factory=ResearchConfig)
    e2e: E2EConfig = field(default_factory=E2EConfig)
    judge: JudgeConfig = field(default_factory=JudgeConfig)
    output: OutputConfig = field(default_factory=OutputConfig)


def load_eval_config(config_path: Optional[str] = None) -> EvalConfig:
    """加载评估配置，自动解析相对路径为绝对路径"""
    if config_path is None:
        config_path = get_abs_path("eval/config.yaml")

    if not os.path.exists(config_path):
        logger.warning(f"[EvalConfig] 配置文件不存在: {config_path}，使用默认配置")
        return EvalConfig()

    with open(config_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    datasets = raw.get("datasets", {})
    for key in datasets:
        datasets[key] = get_abs_path(datasets[key])

    rag = RAGConfig(**raw.get("rag", {}))
    agent = AgentConfig(**raw.get("agent", {}))
    research = ResearchConfig(**raw.get("research", {}))
    e2e = E2EConfig(**raw.get("e2e", {}))
    judge = JudgeConfig(**raw.get("judge", {}))
    output = OutputConfig(**raw.get("output", {}))
    output.reports_dir = get_abs_path(output.reports_dir)

    config = EvalConfig(
        datasets=datasets,
        rag=rag,
        agent=agent,
        research=research,
        e2e=e2e,
        judge=judge,
        output=output,
    )

    logger.info(f"[EvalConfig] 配置加载完成: {config_path}")
    return config