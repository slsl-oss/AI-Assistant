# AI 智能助手 — 基于 STORM 架构的深度研究系统

## 项目简介

基于 **STORM（Synthesis of Topic Outlines through Retrieval and Multi-perspective）** 架构实现的多智能体深度研究系统，集成 RAG 混合检索、长期记忆管理、三级并行研究流水线、可观测性追踪和评估框架，能够根据用户问题自动生成结构化的深度研究报告。

### 技术栈

| 层级 | 技术 | 说明 |
|------|------|------|
| **前端** | HTML/SSE | 用户交互界面，流式输出 |
| **后端** | Java Spring Boot | 用户认证、会话管理、业务编排 |
| **AI 服务** | Python FastAPI | Agent 核心逻辑、RAG、记忆管理 |
| **大模型** | DeepSeek-v4-pro / v4-flash | 核心推理能力（阿里云 DashScope） |
| **向量数据库** | Chroma | 文档、报告与记忆存储 |
| **Agent 框架** | LangChain + LangGraph | Agent 编排与状态图工作流 |
| **评估框架** | eval/ | RAG/Agent/研究/端到端四维评估 |

---

## 系统架构

```
┌──────────────────────────────────────────────────────────────────────┐
│                         前端 (HTML/SSE)                               │
│                SSE 流式输出 / 任务控制 (停止/继续)                      │
└──────────────────────────────────┬───────────────────────────────────┘
                                   │ HTTP/SSE
┌──────────────────────────────────▼───────────────────────────────────┐
│                     Java Spring Boot 后端 (port 8087)                  │
│  ┌──────────────┐  ┌──────────────────┐  ┌────────────────────────┐  │
│  │ UserController│  │SessionController│  │  JWT / SSE 事件推送     │  │
│  └──────────────┘  └──────────────────┘  └────────────────────────┘  │
└──────────────────────────────────┬───────────────────────────────────┘
                                   │ HTTP/SSE
┌──────────────────────────────────▼───────────────────────────────────┐
│                      Python FastAPI 服务 (port 8000)                   │
│                                                                       │
│  ┌─────────────────────────────────────────────────────────────────┐ │
│  │                  Supervisor Agent (主编排)                        │ │
│  │                                                                  │ │
│  │  ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐     │ │
│  │  │ Browser  │──▶│ Planner  │──▶│  Human   │──▶│Researcher│     │ │
│  │  │ (初步调研)│   │ (规划大纲)│   │ (人工审批)│   │ (并行研究)│     │ │
│  │  └──────────┘   └──────────┘   └────┬─────┘   └────┬─────┘     │ │
│  │                      ▲              │ reject        │           │ │
│  │                      └──────────────┘               │           │ │
│  │  ┌──────────┐   ┌──────────┐   ┌──────────┐        │           │ │
│  │  │Publisher │◀──│  Writer  │◀──│Researcher│◀───────┘           │ │
│  │  │ (排版导出)│   │ (引言/结论)│   │ (并行研究)│                    │ │
│  │  └──────────┘   └──────────┘   └──────────┘                    │ │
│  └─────────────────────────────────────────────────────────────────┘ │
│                                                                       │
│  ┌──────────────────────────────┐  ┌──────────────────────────────┐  │
│  │    子编排 (Editor Agent)      │  │   React Agent (通用助手)      │  │
│  │  Researcher → Reviewer →     │  │  日期/天气/搜索/RAG/记忆      │  │
│  │  Reviser (循环, 最多3轮)      │  │  兜底处理简单查询             │  │
│  └──────────────────────────────┘  └──────────────────────────────┘  │
│                                                                       │
│  ┌──────────────────────────────┐  ┌──────────────────────────────┐  │
│  │   三级并行研究流水线           │  │   可观测性 (TraceContext)      │  │
│  │  L1: 子问题并行               │  │  步骤追踪 / 工具计时 / Token  │  │
│  │  L2: 多搜索引擎并行           │  │  错误记录 / 调用链追踪        │  │
│  │  L3: WorkerPool(15) URL抓取   │  │                              │  │
│  └──────────────────────────────┘  └──────────────────────────────┘  │
│                                                                       │
│  ┌──────────────────────────────┐  ┌──────────────────────────────┐  │
│  │  RAG 混合检索                 │  │  长期记忆 (Mem0)              │  │
│  │  Vector + BM25 + RRF         │  │  重要性评分 + 语义检索        │  │
│  │  + BGE Reranker 精排         │  │  自动清理 + 用户隔离          │  │
│  └──────────────────────────────┘  └──────────────────────────────┘  │
└──────────────────────────────────┬───────────────────────────────────┘
                                   │
┌──────────────────────────────────▼───────────────────────────────────┐
│                            存储层                                      │
│  ┌────────────────┐  ┌────────────────┐  ┌──────────────────────────┐│
│  │ Chroma 向量库   │  │ SQLite (Mem0)  │  │ PostgreSQL (Checkpointer) ││
│  │ 文档/报告/记忆  │  │ 记忆索引/历史  │  │ LangGraph 状态持久化      ││
│  └────────────────┘  └────────────────┘  └──────────────────────────┘│
└──────────────────────────────────────────────────────────────────────┘
```

---

## 核心流程

### 1. 查询路由

```
用户输入问题
     │
     ▼
┌─────────────────────┐
│ _is_deep_research() │  ← LLM 分类
└─────────┬───────────┘
          │
    ┌─────┴─────┐
    ▼           ▼
 简单查询    深度研究
    │           │
    ▼           ▼
React Agent  STORM 流水线
(直接回答)   (6节点 DAG)
```

### 2. STORM 深度研究流水线

```
browser → planner → human → researcher → writer → publisher → END
(初步调研) (规划大纲) (审批)  (并行研究)   (引言/结论)  (排版导出)
    │                    │
    │        ┌───────────┘
    │        ▼
    │   子编排 (每个章节)
    │   researcher → reviewer → reviser → reviewer (循环, 最多3轮)
    │
    ▼
向量数据库存储报告 → 未来 RAG 检索可用
```

### 3. 三级并行研究流水线

```
用户查询
    │
    ▼
┌─────────────────────────────────────────────┐
│  L1: 子问题并行 (asyncio.gather)              │
│  查询 → LLM 生成 3-5 个子问题 → 并行执行       │
└────────────────────┬────────────────────────┘
                     │
┌────────────────────▼────────────────────────┐
│  L2: 多搜索引擎并行                           │
│  百度搜索 → URL 去重 → 提取 Top-N URL         │
└────────────────────┬────────────────────────┘
                     │
┌────────────────────▼────────────────────────┐
│  L3: URL 并行抓取 (WorkerPool, Semaphore=15)  │
│  aiohttp + BS4 → 提取正文 → 去噪              │
└────────────────────┬────────────────────────┘
                     │
┌────────────────────▼────────────────────────┐
│  ContextCompressor: Embedding 相似度过滤      │
│  快速通道 (<8000 chars) / 标准管道 (阈值 0.35) │
└────────────────────┬────────────────────────┘
                     │
┌────────────────────▼────────────────────────┐
│  LLM 写草稿 (网页 + RAG 知识库)               │
│  → 合并所有子草稿 → 最终章节草稿               │
└─────────────────────────────────────────────┘
```

### 4. RAG 混合检索流程

```
用户查询
    │
    ▼
┌──────────────────────────────────────────────┐
│  第一阶段: 粗排 (RRF 融合)                     │
│  ┌──────────────┐    ┌──────────────┐        │
│  │ Vector 检索   │    │  BM25 检索    │        │
│  │ (语义相似度)  │    │ (关键词匹配)  │        │
│  └──────┬───────┘    └──────┬───────┘        │
│         └────────┬──────────┘                 │
│                  ▼                            │
│          RRF 融合排序 (高召回)                  │
└──────────────────┬───────────────────────────┘
                   │
┌──────────────────▼───────────────────────────┐
│  第二阶段: 精排 (BGE Reranker)                 │
│  Cross-Encoder 深度语义匹配                    │
│  threshold = max(max_score × 0.3, 0.1)        │
│  → Top-K 结果返回                             │
└──────────────────────────────────────────────┘
```

---

## Agent 清单

| Agent | 角色 | 职责 | 模型 |
|-------|------|------|------|
| **Supervisor** | 主编排 | 查询分类、STORM 流水线调度、上下文压缩 | deepseek-v4-pro |
| **React** | 通用助手 | 日期/天气/搜索/RAG/记忆/兜底对话 | deepseek-v4-flash |
| **Research** | 研究引擎 | 初步调研 + 深度研究（三级并行流水线） | deepseek-v4-pro |
| **Editor** | 子编排 | 规划大纲 + 章节级 researcher→reviewer→reviser 循环 | deepseek-v4-pro |
| **Writer** | 写作 | 引言、结论、目录、来源汇总 | deepseek-v4-pro |
| **Reviewer** | 审查 | 草稿质量四维评估（准确性/完整性/逻辑/清晰度） | deepseek-v4-pro |
| **Reviser** | 修订 | 根据审查意见修改草稿 | deepseek-v4-pro |
| **Publisher** | 排版 | Markdown 报告组装 + 向量数据库存储 | deepseek-v4-flash |
| **Human** | 审批 | 计划审批（模拟人工，可替换为真实人机交互） | deepseek-v4-flash |

---

## 核心功能

### 1. STORM 深度研究

- **两级工作流**：主编排（6 节点 DAG）+ 子编排（review-revise 循环）
- **三级并行**：子问题并行 → 多引擎搜索 → WorkerPool URL 抓取
- **上下文压缩**：Embedding 相似度过滤（快速通道 + 标准管道）
- **RAG 增强**：网页抓取 + 本地知识库双源输入 LLM
- **报告存储**：生成的报告自动存入向量数据库，支持后续检索

### 2. RAG 混合检索

| 特性 | 说明 |
|------|------|
| **混合检索** | Vector (语义) + BM25 (关键词) + RRF 融合 |
| **两阶段检索** | 粗排 (RRF) → 精排 (BGE Cross-Encoder) |
| **动态阈值** | `threshold = max(max_score × ratio, abs_threshold)` |
| **多格式支持** | TXT, PDF, DOCX, PPTX, XLSX, HTML, CSV, JSON, XML, MD, 图片, 音频 |
| **Markdown 分块** | 按标题层级智能分块，保持语义边界 |

### 3. 长期记忆 (Mem0)

| 功能 | 说明 |
|------|------|
| **自动事实提取** | 10 轮对话后自动提取关键事实 |
| **重要性评分** | LLM 评分 0-1，区分高/低价值记忆 |
| **语义检索** | 基于 Chroma 的语义记忆搜索 |
| **清理策略** | 低重要性 / 过期 (>30天) / 超容量 (>100条) 三层清理 |
| **用户隔离** | 支持用户专属记忆与共享记忆 |

### 4. 可观测性

| 维度 | 内容 |
|------|------|
| **调用链路** | TraceContext: trace_id, agent_path, steps |
| **工具调用** | 名称、参数、成功/失败、耗时 |
| **Token 估算** | 输入/输出 token 估算（中英文混合算法） |
| **错误追踪** | 失败节点、错误信息 |
| **响应时间** | 总耗时 + 各阶段耗时 |

### 5. 评估框架 (eval/)

| 维度 | 指标 | 命令 |
|------|------|------|
| **RAG 检索** | Hit Rate@k, MRR, NDCG@k, Reranker 对比 | `python -m eval.cli rag` |
| **Agent 回答** | 工具选择准确率, LLM Judge 四维评分 | `python -m eval.cli agent` |
| **研究报告** | 章节覆盖率, 来源引用, 事实准确性 | `python -m eval.cli research` |
| **端到端** | 延迟分布(p50/p95/p99), Token, 成本 | `python -m eval.cli e2e` |

---

## 目录结构

```
AI assistant/
├── agent/                          # Agent 核心模块
│   ├── supervisor_agent.py         # 主编排 (STORM 流水线)
│   ├── react_agent.py              # 通用助手 (ReAct)
│   ├── research_agent.py           # 研究引擎 (初步调研 + 深度研究)
│   ├── editor_agent.py             # 子编排 (planner + review-revise)
│   ├── writer_agent.py             # 写引言/结论
│   ├── reviewer_agent.py           # 审查草稿质量
│   ├── reviser_agent.py            # 修订草稿
│   ├── publisher_agent.py          # 排版导出 + 报告存储
│   ├── human_agent.py              # 人工审批 (模拟)
│   ├── infrastructure/             # 基础设施
│   │   ├── research_pipeline.py    # 三级并行研究流水线
│   │   ├── web_scraper.py          # BS4 + WorkerPool 网页抓取
│   │   └── context_compressor.py   # Embedding 上下文压缩
│   └── tools/                      # Agent 工具
│       ├── base_agent_tools.py     # 搜索/RAG/记忆/日期/天气工具
│       └── middleware.py           # 工具监控 + 可观测性 + 提示词切换
├── rag/                            # RAG 检索模块
│   ├── vector_stores.py            # Chroma + BM25 + RRF + Reranker
│   ├── reranker.py                 # BGE Cross-Encoder 精排
│   ├── markdown_chunker.py         # Markdown 智能分块
│   ├── rag_service.py              # RAG 摘要服务
│   └── upload_service.py           # 文档上传处理
├── memory/                         # 长期记忆模块
│   ├── mem0_service.py             # Mem0 服务封装
│   ├── memory_tools.py             # 记忆工具 (save/search)
│   ├── memory_scorer.py            # LLM 重要性评分
│   ├── memory_index.py             # SQLite 记忆索引
│   └── memory_cleanup.py           # 清理策略 (三层)
├── eval/                           # 评估框架
│   ├── cli.py                      # CLI 入口
│   ├── reporter.py                 # JSON + Markdown 报告
│   ├── config.yaml                 # 评估配置
│   ├── metrics/                    # 指标计算 (纯函数)
│   │   ├── rag_metrics.py          # Hit Rate, MRR, NDCG, Precision, Recall
│   │   ├── agent_metrics.py        # 工具准确率, LLM Judge 评分
│   │   ├── research_metrics.py     # 覆盖率, 引用率, 准确性
│   │   └── e2e_metrics.py          # 延迟分布, Token, 成本
│   ├── datasets/                   # 测试数据集
│   │   ├── rag_test_queries.json   # 15 条 RAG 查询
│   │   ├── agent_test_tasks.json   # 12 条 Agent 任务
│   │   └── research_topics.json    # 3 个研究主题
│   ├── runners/                    # 评估执行器
│   │   ├── rag_runner.py           # 对接 VectorStoreService
│   │   ├── agent_runner.py         # 对接 SupervisorAgent
│   │   ├── research_runner.py      # 执行 STORM 流水线
│   │   └── e2e_runner.py           # 延迟/Token/成本基准
│   └── judges/                     # LLM-as-Judge
│       ├── llm_judge.py            # 四维评分 + 事实准确性
│       └── prompts/                # 评分 prompt 模板
├── model/                          # 模型工厂
│   └── factory.py                  # ChatTongyi + DashScopeEmbeddings
├── utils/                          # 工具函数
│   ├── config_handler.py           # YAML 配置加载
│   ├── prompts_loader.py           # Prompt 文件加载
│   ├── logger_handler.py           # 日志管理
│   ├── observability.py            # TraceContext 可观测性
│   ├── postgres_checkpointer.py    # PostgreSQL 检查点持久化
│   ├── cancellation.py             # 协作式取消令牌
│   └── file_handler.py             # 文件处理 (MD5/加载)
├── config/                         # YAML 配置
│   ├── agent.yaml                  # Agent 模型映射 + DB URI
│   ├── chroma.yaml                 # 向量库 + Reranker 配置
│   ├── memory.yaml                 # Mem0 记忆配置
│   ├── prompts.yaml                # Prompt 文件路径映射
│   └── rag.yaml                    # RAG 模型配置
├── prompts/                        # Prompt 模板
│   ├── supervisor_agent_prompt.txt # 主编排路由规则
│   ├── react_agent_prompt.txt      # 通用助手决策流程
│   ├── research_agent_prompt.txt   # 研究 Agent 指令
│   └── rag_summarize.txt           # RAG 摘要模板
├── agent_service/                  # Java Spring Boot 后端
│   └── src/main/java/...           # Controller, Service, Mapper, Config
├── server.py                       # Python FastAPI 入口
└── README.md
```

---

## 快速开始

### 环境要求

- Python 3.10+
- Java 17+
- PostgreSQL 14+（可选，用于生产环境 Checkpointer）
- 阿里云 DashScope API Key

### 安装依赖

```bash
# Python 依赖
pip install langchain langgraph langchain-community langchain-chroma
pip install sentence-transformers
pip install mem0ai
pip install fastapi uvicorn aiohttp beautifulsoup4 lxml
pip install PyYAML

# Java 依赖 (Maven)
cd agent_service && mvn install
```

### 配置

```bash
# 设置环境变量
export DASHSCOPE_API_KEY="your-api-key"

# 或创建 .env 文件
echo "DASHSCOPE_API_KEY=your-api-key" > .env
```

### 启动服务

```bash
# 启动 Python AI 服务
python server.py

# 启动 Java 后端服务
cd agent_service && mvn spring-boot:run
```

### 运行评估

```bash
# 全量评估
python -m eval.cli all

# 快速评估（减少迭代次数）
python -m eval.cli all --quick

# 单项评估
python -m eval.cli rag       # RAG 检索质量
python -m eval.cli agent     # Agent 回答质量
python -m eval.cli research  # 研究报告质量
python -m eval.cli e2e       # 端到端性能
```

---

## 配置说明

### agent.yaml — Agent 模型映射

```yaml
chat_model_name: deepseek-v4-pro
agent_models:
  supervisor:      deepseek-v4-pro
  react_agent:     deepseek-v4-flash
  research_agent:  deepseek-v4-pro
  editor:          deepseek-v4-pro
  writer:          deepseek-v4-pro
  reviewer:        deepseek-v4-pro
  reviser:         deepseek-v4-pro
  publisher:       deepseek-v4-flash
  human:           deepseek-v4-flash
```

### chroma.yaml — RAG 检索配置

```yaml
collection_name: agent
persist_directory: chroma_db
k: 5
rerank_enabled: true
coarse_k_multiplier: 3
rerank_top_k: 3
rerank_score_ratio: 0.3
rerank_abs_threshold: 0.1
```

### memory.yaml — 长期记忆配置

```yaml
collection_name: long_term_memory
chroma_path: memory_db
k: 5
memory_limit_per_user: 100
importance_threshold: 0.3
retention_days: 30
```

---

## 项目亮点

1. **STORM 架构**：两级 LangGraph 工作流，生成结构化深度研究报告
2. **三级并行研究**：子问题并行 → 多引擎搜索 → WorkerPool 抓取，最大化吞吐
3. **两阶段 RAG**：RRF 粗排 + BGE Reranker 精排，兼顾召回与精度
4. **RAG 增强研究**：网页抓取 + 本地知识库双源输入，报告自动存入向量库
5. **可观测性**：全链路 TraceContext 追踪，工具计时、Token 估算、错误定位
6. **评估框架**：四维自动化评估，LLM Judge 评分，JSON/Markdown 报告
7. **任务中断恢复**：PostgreSQL Checkpointer 持久化状态，随时停止/继续
8. **零侵入评估**：eval/ 目录完全独立，无需修改现有代码