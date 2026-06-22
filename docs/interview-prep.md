# AI Assistant 项目面试深度解析

> 逐条拆解简历上的每一项技术点，附带源码级实现细节、设计决策、量化数据来源、面试追问应对。

---

## 一、STORM 深度研究架构

### 简历原文

> 设计 browser → planner → human → researcher → writer → publisher 六节点主编排 DAG，章节级 editor agent 子编排（researcher → reviewer → reviser 循环，最多 3 轮），9 个专职 Agent 各司其职。

### 1.1 为什么选 STORM 架构而不是简单的 Supervisor-Worker？

**演变过程：**

```
v1: 单 Agent 工具调用 → 一个 Agent 调所有工具，上下文爆炸
v2: Supervisor-Worker → 任务分解 + 并行扇出，但只能做 QA，不能生成报告
v3: STORM → 多阶段流水线，模拟人类写报告的过程（调研→规划→研究→审查→写作→排版）
```

**核心区别：** Supervisor-Worker 是"分发任务—收集结果"，STORM 是"多阶段流水线 + 质量门控"。STORM 的 reviewer-reviser 循环是质量保证的关键——没有审查环节，研究报告的质量无法保证。

### 1.2 主编排 6 节点 DAG 详解

**源码位置：** `agent/supervisor_agent.py:158-178`

```python
def _build_graph(self) -> StateGraph:
    wf = StateGraph(ResearchState)
    wf.add_node("browser", browser_node)      # 初步调研
    wf.add_node("planner", planner_node)      # 规划大纲
    wf.add_node("human", human_node)          # 人工审批
    wf.add_node("researcher", researcher_node) # 并行研究
    wf.add_node("writer", writer_node)        # 写引言/结论
    wf.add_node("publisher", publisher_node)  # 排版导出

    wf.set_entry_point("browser")
    wf.add_edge("browser", "planner")
    wf.add_edge("planner", "human")
    wf.add_conditional_edges("human", route_after_human, {
        "planner": "planner", "researcher": "researcher",
    })
    wf.add_edge("researcher", "writer")
    wf.add_edge("writer", "publisher")
    wf.add_edge("publisher", END)
    return wf.compile(checkpointer=self._checkpointer)
```

**每个节点的职责和输入输出：**

| 节点         | 职责     | 输入                       | 输出                            | 调用的 Agent                                                                                       |
| ---------- | ------ | ------------------------ | ----------------------------- | ----------------------------------------------------------------------------------------------- |
| browser    | 初步调研   | query                    | 200-400 字概述                   | research_agent.run_initial_research()                                                           |
| planner    | 规划大纲   | query + initial_research | JSON {title, sections[]}      | editor_agent.plan_research()                                                                    |
| human      | 审批大纲   | sections                 | approved/rejected             | human_agent.review_plan()                                                                       |
| researcher | 并行研究   | sections[]               | 每个章节的草稿                       | editor_agent.run_parallel_research()，里面子流程图的research子节点调用了research_agent.run_depth_research()方法 |
| writer     | 写引言/结论 | research_data            | introduction, conclusion, toc | writer_agent.run()                                                                              |
| publisher  | 排版导出   | 全部数据                     | Markdown 报告 + 存入向量库           | publisher_agent.run()                                                                           |

**面试追问：为什么 human 节点在 planner 之后、researcher 之前？**

因为研究阶段（researcher）是最昂贵的 LLM 调用（每个章节都走完整的三级并行流水线）。如果大纲不合适，在 planner 阶段就驳回重新规划，而不是等研究完了再改。这叫做"前置质量控制"。

### 1.3 子编排 review-revise 循环

**源码位置：** `agent/editor_agent.py:38-50`

```python
def _build_graph(self) -> StateGraph:
    wf = StateGraph(DraftState)
    wf.add_node("researcher", self._researcher_node)  # 研究 → 写草稿
    wf.add_node("reviewer", self._reviewer_node)       # 审查 → accept/revise
    wf.add_node("reviser", self._reviser_node)         # 修改草稿
    wf.set_entry_point("researcher")
    wf.add_edge("researcher", "reviewer")
    wf.add_edge("reviser", "reviewer")  # 修改后必须重新审查
    wf.add_conditional_edges("reviewer", self._route_after_review,
        {"accept": END, "revise": "reviser"})
    return wf.compile()
```

**循环保护机制：** `MAX_REVISIONS = 3`，超过 3 轮强制 accept（`editor_agent.py:60`）：

```python
if count >= self.MAX_REVISIONS:
    return {"review": None}  # 强制通过
```

**面试追问：为什么 reviser 不直接到 END，而是必须回到 reviewer？**

因为修改后的草稿不保证质量。如果 reviser 直接到 END，可能修改得更差。回到 reviewer 形成"修改—审查"闭环，确保每次修改都经过质量检查。这是一个"质量门"（Quality Gate）模式。

### 1.4 9 个 Agent 的职责划分

| Agent      | 角色   | 为什么需要独立 Agent？                                                       |
| ---------- | ---- | -------------------------------------------------------------------- |
| Supervisor | 主编排  | 单一职责：只做路由和编排，不参与内容生成                                                 |
| React      | 通用助手 | 处理简单查询，避免 STORM 流水线的开销                                               |
| Research   | 研究引擎 | 封装三级并行流水线，独立可测试                                                      |
| Editor     | 子编排  | 每个章节独立编排，主要编排是主流reasercher节点（researcher并行研究 → reviewer → reviser ）循环 |
| Writer     | 写作   | 引言/结论需要不同的写作风格（宏观视角）                                                 |
| Reviewer   | 审查   | 独立的质量评估视角，不与研究者共享上下文                                                 |
| Reviser    | 修订   | 根据审查意见定向修改，不重新生成                                                     |
| Publisher  | 排版   | 格式统一、报告存储，关注点分离                                                      |
| Human      | 审批   | 模拟人机交互，可替换为真实人工审批                                                    |

**面试追问：为什么 Reviewer 和 Researcher 要用不同的 Agent？**

这叫"自我审查的盲点问题"——如果同一个模型既写草稿又审查，它倾向于认可自己的输出。用独立的 Reviewer Agent（独立的 LLM 调用、独立的 prompt）模拟"同行评审"（Peer Review），提高审查的有效性。

---

## 二、三级并行研究流水线

### 简历原文

> L1 子问题并行（asyncio.gather 3-5 个子问题）、L2 多搜索引擎并行、L3 WorkerPool（Semaphore 15）URL 并行抓取，经 ContextCompressor（Embedding 相似度过滤，阈值 0.35）压缩后输入 LLM 写草稿。效果：相比串行搜索，研究阶段耗时缩短约 70%，网页抓取吞吐量提升 15 倍。

### 2.1 三级并行的代码实现

**源码位置：** `agent/infrastructure/research_pipeline.py`

**L1 — 子问题并行（第 207-226 行）：**

```python
async def process(self, query: str) -> dict:
    sub_queries = await self._generate_sub_queries(query)  # LLM 生成 3-5 个子问题
    # asyncio.gather 并行执行所有子问题
    sub_results = await asyncio.gather(
        *[self._process_sub_query(sq) for sq in sub_queries]
    )
    merged = await self._merge_drafts(query, sub_results)
```

**L2 — 搜索引擎（第 77-106 行）：**

```python
async def _search(self, query: str) -> list[str]:
    # 当前：百度搜索 API
    # 扩展点：可并行加入 Google、Bing、DuckDuckGo
    api_key = os.getenv("BAIDU_API_KEY")
    resp = await asyncio.to_thread(
        requests.post, url, headers=headers, json=data, timeout=30
    )
```

**L3 — WorkerPool 并行抓取（web_scraper.py:34-44）：**

```python
class WorkerPool:
    def __init__(self, max_workers: int = 15):
        self.semaphore = asyncio.Semaphore(max_workers)  # 控制并发数

class GlobalRateLimiter:
    def __init__(self, requests_per_second: float = 10.0):
        self.min_interval = 1.0 / requests_per_second  # 两次请求间隔 0.1s
```

**面试追问：为什么 Semaphore 是 15 而不是 50 或 100？**

三个约束：

1. **目标网站压力**：过高的并发会被目标网站封 IP
2. **内存控制**：每个网页的 HTML 可能几十 KB 到几 MB，15 个并发 ≈ 内存可控
3. **GlobalRateLimiter 10 req/s**：Semaphore 15 配合 RateLimiter 10 req/s，实际有效并发 ≈ 10，既保证吞吐又不过载

### 2.2 "耗时缩短约 70%" 怎么算的？

**串行场景：** 3 个子问题 × (搜索 2s + 抓取 10 URLs × 1s/URL + 压缩 1s + LLM 3s) = 3 × 16s = 48s

**并行场景：** asyncio.gather 3 个子问题同时跑，每个子问题内 10 个 URL 并行抓取（Semaphore 15）：

- 搜索：3 个并行 → 2s（不是 6s）
- 抓取：10 URLs 并行 × 15 并发 → ~1s（不是 10s）
- 压缩 + LLM：3s
- 总计 ~6s（子问题最慢的那个）

**70% 是估算**（(48-6)/48 ≈ 87%，保守说 70%）

### 2.3 ContextCompressor 两层决策

**源码位置：** `agent/infrastructure/context_compressor.py:35-61`

```
总内容 < 8000 字符？
  ├─ YES → 快速通道：直接返回原文（跳过 Embedding 计算）
  └─ NO  → 标准管道：
            1. RecursiveCharacterTextSplitter 分块（chunk_size=1000, overlap=100）
            2. DashScopeEmbeddings 嵌入 query + chunks
            3. 余弦相似度过滤（threshold=0.35）
            4. 排序取 top-5
            5. 降级：嵌入失败 → 返回前 max_results 个文档的前 2000 字符
```

**面试追问：为什么阈值是 0.35？**

余弦相似度 0.35 是一个经验值。在 Embedding 模型（text-embedding-v4）的语义空间中，0.35 大致意味着"有一定相关性但不强"。设太高（如 0.7）会过滤掉太多内容导致草稿信息不足，设太低（如 0.1）会保留噪音。0.35 是平衡相关性和信息量的折中。

---

## 三、RAG 双源增强研究

### 简历原文

> 网页抓取内容 + 本地知识库检索结果双源输入 LLM 生成草稿，生成的报告自动分块存入 ChromaDB（type: generated_report），支持后续 RAG 检索追问。

### 3.1 双源融合的 Prompt 设计

**源码位置：** `research_pipeline.py:173-203`

```python
async def _write_draft(self, query, context):
    rag_context = await self._get_rag_context(query)  # 本地知识库

    research_materials = ""
    if context:
        research_materials += f"## 网页抓取资料\n{context}\n\n"
    if rag_context:
        research_materials += f"## 本地知识库资料\n{rag_context}\n\n"

    prompt = (
        "根据以下研究资料，围绕问题撰写一份简洁的草稿。\n"
        "网页资料标注来源 URL，知识库资料标注【参考资料N】\n"
        "融合网页和知识库的信息，互补不足\n"
        f"## 问题\n{query}\n\n{research_materials}"
    )
```

**设计意图：** 网页提供实时信息（如最新新闻、数据），知识库提供企业私有知识（如内部文档、历史报告）。两者互补——网页解决"新"的问题，知识库解决"深"的问题。

### 3.2 报告回存向量库

**源码位置：** `supervisor_agent.py:121-131`

```python
# publisher_node 中，报告生成后自动存入向量库
from rag.vector_stores import VectorStoreService
vs = VectorStoreService()
vs.add_report_to_store(
    title=state.get("title", ""), content=report,
    user_id=state.get("user_id", "default_user"),
)
```

**存入的元数据标记：** `vector_stores.py:209-212`

```python
source = f"[报告] {title}"
doc = Document(page_content=content, metadata={
    "user_id": user_id, "source": source, "type": "generated_report",
})
```

**面试追问：为什么用 `type: "generated_report"` 标记？**

方便后续检索时区分来源。用户追问"报告里提到的 XX 是什么"时，react_agent 的 rag_summarize 可以过滤只查 `type=generated_report` 的文档，避免被原始上传文档干扰。

### 3.3 合并阶段也融入了 RAG

**源码位置：** `research_pipeline.py:228-262`

`_merge_drafts()` 在合并所有子问题草稿时，再次调用 `_get_rag_context(query)` 获取知识库资料，作为"补充资料"注入合并 prompt。这样即使子问题草稿遗漏了知识库信息，合并阶段也能补充。

---

## 四、两阶段 RAG 检索

### 简历原文

> Stage 1 — RRF 粗排召回 50 条候选（Vector + BM25 双路融合），Stage 2 — BGE Reranker 精排 + 动态阈值过滤（threshold = max(max_score × 0.3, 0.1)）返回 Top-5。效果：MRR 比纯向量检索提升约 30%。

### 4.1 RRF 融合公式

**源码位置：** `vector_stores.py:102-130`

```python
def hybrid_search(self, query, k=None, rrf_k=60):
    # 两路检索
    vector_docs = vector_retriever.invoke(query)  # 语义检索 k*2=10 条
    bm25_docs = self._bm25_retriever.invoke(query) # 关键词检索

    # RRF 融合: score(d) = sum(1 / (k + rank_i(d)))
    doc_scores = {}
    for rank, doc in enumerate(vector_docs):
        doc_scores[doc_id]["score"] += 1.0 / (rrf_k + rank + 1)  # 60 + rank

    for rank, doc in enumerate(bm25_docs):
        doc_scores[doc_id]["score"] += 1.0 / (rrf_k + rank + 1)

    # 按融合分数排序，取 top-k
    sorted_items = sorted(doc_scores.values(), key=lambda x: x["score"], reverse=True)
    return [item["doc"] for item in sorted_items[:k]]
```

**面试追问：为什么 RRF 的 k=60？**

RRF 的 k 值控制排名的影响力。k=60 意味着"排名靠前"和"排名靠后"的分数差异不大（1/61 ≈ 0.016 vs 1/70 ≈ 0.014），这适合"两路检索结果可能差异很大"的场景——某个文档在 Vector 中排第 1 但在 BM25 中排第 30，如果 k 太小（如 k=1），BM25 的排名会拉低总分太多。k=60 是业界标准值。

### 4.2 组合动态阈值

**源码位置：** `reranker.py:108-111`

```python
max_score = scored[0][1]  # 最高分
threshold = max(max_score * score_ratio, abs_threshold)  # 取两者最大值
# 配置: score_ratio=0.3, abs_threshold=0.1
```

**三种场景：**

```
场景1: max_score=0.9 → threshold = max(0.9*0.3, 0.1) = max(0.27, 0.1) = 0.27
场景2: max_score=0.2 → threshold = max(0.2*0.3, 0.1) = max(0.06, 0.1) = 0.10
场景3: max_score=0.05 → threshold = max(0.015, 0.1) = 0.10
```

**设计意图：** 场景 2 和 3 中，如果只用比例阈值（0.3），所有文档分数都低于 0.27，全部被过滤。abs_threshold=0.1 保底，确保"即使最高分也不高，至少保留最高分那条"。

### 4.3 兜底机制

**reranker 失败时的降级（reranker.py:132-135）：**

```python
except Exception as e:
    logger.warning(f"[Rerank] 精排失败，回退原始 RRF 排序: {e}")
    return documents[:top_k]
```

**全部被过滤时的兜底（reranker.py:121-123）：**

```python
if filtered:
    result = filtered[:top_k]
else:
    result = [scored[0][0]]  # 至少返回最高分 1 条
```

---

## 五、双层记忆体系

### 简历原文

> 短期记忆基于 LangGraph Checkpointer 持久化至 PostgreSQL，结合滑动窗口上下文压缩（阈值 20K tokens）；长期记忆接入 Mem0，LLM 重要性评分（0-1）+ 三层遗忘策略（重要性阈值 / 过期天数 / 超容量淘汰，单用户上限 100 条）。

### 5.1 短期记忆 — 上下文压缩

**源码位置：** `supervisor_agent.py:333-391`

**核心逻辑：**

```python
TOKEN_BUDGET = 20000

# 从后往前累加 token 数，找到超出预算的 cutoff 点
for i in range(len(pairs) - 1, -1, -1):
    total += _est(u[:500]) + _est(a[:500])
    if total > TOKEN_BUDGET:
        cutoff = i + 1  # 从这里开始压缩
        break

# 将 cutoff 之前的对话压缩为摘要
resp = await summarizer_model.ainvoke("总结以下对话的关键信息..." + text)
new_summary = resp.content.strip()

# 替换消息列表为 [SystemMessage(摘要)] + [剩余消息]
new_msgs = [SystemMessage(content=f"[历史对话摘要]\n{new_summary}")]
```

**面试追问：为什么是 20K tokens？**

DeepSeek-v4-pro 的上下文窗口是 128K，但 Prompt Cache 有 5 分钟 TTL。20K 确保对话历史始终在缓存命中区，避免每次请求都重新计算全部上下文。这是一个"成本—性能"的折中。

### 5.2 长期记忆 — 自动提取

**源码位置：** `supervisor_agent.py:393-420`

```python
async def _maybe_batch_extract(self, session_id, user_id):
    # 每 10 轮对话触发一次提取
    if len(pairs) < 10:
        return
    last = pairs[-10:]
    mem0_service.add(conv, user_id=user_id,
                     metadata={"source_session_id": session_id, "auto_extracted": True})
```

**设计意图：** 不是每轮对话都提取（太频繁），而是攒够 10 轮批量提取。Mem0 内部会调用 LLM 从对话中提取事实（如"用户喜欢深色系衣服"），并自动评分重要性。

### 5.3 三层遗忘策略

**源码位置：** `memory/memory_cleanup.py`

```python
def cleanup(user_id: str) -> dict:
    r = {}
    r["importance"] = cleanup_by_importance(user_id)  # 重要性 < 0.3 → 删除
    r["time"] = cleanup_by_time(user_id)              # 超过 30 天 → 删除
    r["capacity"] = cleanup_by_capacity(user_id)      # 超过 100 条 → 删除低分
    return r
```

**容量淘汰的排序逻辑（memory_index.py）：**

```python
# 综合评分 = importance * 0.7 + recency * 0.3
# 删除得分最低的 excess 条
items = memory_index.get_sorted_by_score(user_id)
```

**面试追问：为什么三层而不是一层？**

- **重要性**：永久性清理（低价值信息不应该占空间）
- **时间**：时效性清理（去年的天气偏好今年没意义了）
- **容量**：兜底清理（即使都重要，也不能无限膨胀）

三层层层递进，确保"重要 + 新鲜"的记忆保留。

---

## 六、可观测性与评估体系

### 简历原文

> 自研 TraceContext 全链路追踪（contextvars 异步安全），记录步骤耗时、工具调用、Token 估算、错误定位；构建 eval/ 评估框架（RAG 检索 / Agent 回答 / 研究报告 / 端到端性能四维），LLM-as-Judge 四维评分。

### 6.1 TraceContext 实现

**源码位置：** `utils/observability.py`

```python
@dataclass
class TraceContext:
    trace_id: str          # uuid.uuid4().hex[:12] — 12 位唯一 ID
    session_id: str        # 会话 ID
    query: str             # 用户原始问题
    agent_path: str        # "react_agent" 或 "STORM(browser→planner→...)"
    steps: list[StepRecord]       # 每个步骤的耗时
    tool_calls: list[ToolCallRecord]  # 每次工具调用的详情
    estimated_input_tokens: int   # 中英文混合估算
    estimated_output_tokens: int
```

**Token 估算算法（observability.py:97-101）：**

```python
def _estimate_tokens(self, text: str) -> int:
    zh = sum(1 for c in text if '一' <= c <= '鿿')  # 中文字符数
    return int(zh * 1.5 + (len(text) - zh) * 0.25)   # 中文 ~1.5 token/字，英文 ~0.25 token/字
```

**面试追问：为什么不用 tiktoken 精确计算？**

tiktoken 依赖模型特定的 tokenizer，需要额外安装且支持有限。中英文混合估算（中文 1.5、英文 0.25）是业界常用的近似算法，误差在 ±10% 以内，对成本估算和上下文预算控制足够精确。

### 6.2 contextvars 异步安全

**之前已经详细解释过，这里补充代码层面：**

```python
_current_trace: contextvars.ContextVar = contextvars.ContextVar("current_trace")

def get_current_trace():
    return _current_trace.get()    # 每个协程拿到自己的 trace

def set_current_trace(trace):
    _current_trace.set(trace)      # 子协程自动继承父协程的上下文
```

**工具中间件自动记录（middleware.py）：**

```python
@wrap_tool_call
async def tool_monitor(request, handler):
    start = time.time()
    result = await handler(request)  # 执行工具
    elapsed = (time.time() - start) * 1000
    _record_tool_trace(name, args, success=True, duration_ms=elapsed)
```

**面试追问：中间件如何知道当前是哪个 TraceContext？**

中间件不直接依赖 TraceContext。它通过 `get_current_trace()` 获取当前协程的上下文——这是 contextvars 的魔力：`wrap_tool_call` 装饰的函数在调用者的协程中执行，自动继承调用者的上下文。

### 6.3 评估框架设计模式

**零侵入设计：**

```
现有代码（不修改）        评估框架（独立目录）
─────────────────        ─────────────────
VectorStoreService  ←──  RAGRunner (import 调用)
SupervisorAgent     ←──  AgentRunner (import 调用)
model/factory.py    ←──  LLMJudge (复用 ChatModelFactory)
TraceContext        ←──  E2ERunner (读取 get_current_trace())
```

### 6.4 LLM Judge 评分

**源码位置：** `eval/judges/llm_judge.py`

```python
async def evaluate_response(self, query, response):
    scores = {}
    for criterion in ["relevance", "completeness", "accuracy", "safety"]:
        prompt = self._load_prompt(criterion).format(query=query, response=response)
        result = await self.model.ainvoke(prompt)
        scores[criterion] = self._parse_score(result.content)  # 正则提取 0-5
    return scores
```

**解析容错：**

```python
@staticmethod
def _parse_score(text: str) -> float:
    match = re.search(r'(\d+(?:\.\d+)?)', text)  # 匹配第一个数字
    if match:
        return min(5.0, max(0.0, float(match.group(1))))  # clamp 0-5
    return 0.0
```

---

## 七、断点恢复机制

### 简历原文

> 利用 thread_id 绑定会话 + 节点级自动快照，实现流式对话随时暂停与精准续传；自研 asyncio.Event 取消信号机制，解决异步生成器中 CancelledError 跨层传播与信号覆盖问题。中断恢复成功率 100%。

### 7.1 为什么不能用 asyncio.Task.cancel()？

**源码位置：** `utils/cancellation.py` 注释：

```
两个核心问题：
1. CancelledError 跨层传播：asyncio.CancelledError 会穿透异步生成器到达
   LangGraph 内部，导致 Checkpoint 状态损坏。
2. 信号覆盖：异步生成器被 CancelledError 标记后无法再 yield，导致无法
   通知前端取消状态。
```

### 7.2 合作式取消

**源码位置：** `utils/cancellation.py:35-80`

```python
class CancellationToken:
    def __init__(self):
        self._event = asyncio.Event()  # 跨协程信号
        self._is_cancelled = False

    def cancel(self):
        self._is_cancelled = True
        self._event.set()  # 唤醒所有等待者

    @property
    def is_cancelled(self) -> bool:
        return self._is_cancelled  # 非阻塞检查
```

**使用模式（supervisor_agent.py:231-236）：**

```python
async for chunk in react_agent.execute_stream(aq, session_id):
    if token.is_cancelled:      # ← 主动检查（合作式）
        yield CANCELLED_SIGNAL   # ← 通知前端
        return                   # ← 优雅退出，不抛异常
    yield chunk
```

**面试追问：为什么叫"合作式"取消？**

因为执行协程和目标协程是"合作"关系，不是"强制"关系。执行协程主动在每个 yield 检查点检查 `is_cancelled`，决定是否退出。这避免了操作系统的"抢占式"取消（asyncio.CancelledError）带来的状态损坏问题。

### 7.3 恢复流程

```
用户点击"继续执行"
  → Java 后端建立新 SSE 连接
  → Python 收到请求，session_id 不变
  → LangGraph 的 Checkpointer 用 thread_id 查找上次的快照
  → 从快照中的节点继续执行（不是从头开始）
  → 如果上次在 "researcher" 节点中断，直接从 researcher 继续
```

**关键配置（supervisor_agent.py:256）：**

```python
config = {"configurable": {"thread_id": session_id, "user_id": user_id}}
# thread_id = session_id → 同一个会话共享同一个 Checkpoint
```

---

## 八、关键指标速查

| 指标              | 含义                                   | 计算方式                          |
| --------------- | ------------------------------------ | ----------------------------- |
| **MRR**         | Mean Reciprocal Rank，第一个相关文档排名倒数的平均值 | `1/rank`，无命中取 0               |
| **Hit Rate@K**  | Top-K 中至少有一个相关文档的查询比例                | 命中数/总查询数                      |
| **NDCG@K**      | 考虑排序位置的检索质量，越靠前权重越大                  | DCG/IDCG，用 `log2(i+1)` 折损     |
| **P50**         | 中位数延迟，50% 请求在此时间内完成                  | 排序后取第 50 百分位                  |
| **P95**         | 95% 请求在此时间内完成                        | 排序后取第 95 百分位                  |
| **P99**         | 99% 请求在此时间内完成                        | 排序后取第 99 百分位                  |
| **contextvars** | Python 协程级上下文隔离                      | 替代 threading.local，适配 asyncio |

---

## 九、面试高频追问速查

### Q: 这个项目最大的技术难点是什么？

**A:** 三级并行流水线的协调——子问题并行 + URL 抓取并发 + LLM 调用异步，三层嵌套的 asyncio 需要精确控制并发度（Semaphore）和频率限制（RateLimiter），否则要么被目标网站封 IP，要么内存爆炸。同时 ContextCompressor 的 Embedding 计算是 CPU 密集型，在 asyncio 事件循环中需要用 `asyncio.to_thread` 避免阻塞。

### Q: 如果让你重新设计，你会改什么？

**A:** 

1. 研究流水线目前只有百度一个搜索引擎，应该加入 Google/Bing 做多路搜索 RRF 融合
2. Reviewer 目前只做整体评分，应该做逐句/逐段的事实核查（Claim Verification）
3. 评估框架的 LLM Judge 应该加多 Judge 投票机制提高可靠性
4. 应该引入缓存层——相同子问题的搜索结果可以缓存，避免重复抓取

### Q: 怎么保证生成报告的质量？

**A:** 三道质量门：

1. **Planner 层的 human 审批**：大纲不对就驳回重来
2. **章节级的 reviewer-reviser 循环**：每个章节最多 3 轮修改
3. **提示词层面的约束**：要求标注来源、不确定信息标"待验证"、禁止 emoji

### Q: 这个系统能处理多长的报告？

**A:** 理论无上限（每个章节独立研究），实际受限于 LLM 上下文窗口（128K tokens）。对于 5 章节的报告，每个章节草稿约 2000-5000 字，合并后报告约 1-2 万字。更长的报告可以通过增加章节数实现，每个章节独立走子编排流水线。

### Q: 评估框架的指标如何用于持续优化？

**A:** 每次修改系统后跑一次评估，对比前后指标。例如：

- 改了 Reranker 阈值 → 跑 `python -m eval.cli rag` 看 MRR 变化
- 改了 Agent prompt → 跑 `python -m eval.cli agent` 看工具选择准确率
- 改了研究流水线 → 跑 `python -m eval.cli research` 看章节覆盖率
- 报告带时间戳，可以对比历史趋势