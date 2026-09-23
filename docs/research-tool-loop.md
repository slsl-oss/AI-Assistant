# Research Agent 三工具受控循环（离线框架）

## 实现与边界

外层保留 LangGraph 主图和章节子图，章节 Researcher 内新增 `ResearchLoop`：
决策、校验、执行工具、把实际结果作为 observation 返回、再次决策。它是有预算的
反馈循环，不是预排好的 search/fetch/write 队列。原 `ResearchPipeline` 默认保留；
显式注入 `research_loop` 才启用新路径。

**没有接真实模型、搜索 API、知识库或网页抓取服务。** 模拟策略的 if/else 仅用于
验证框架，不是 LLM 推理或准确率评测。演示中的产品、数字与 example.org 页面
全是虚构资料，不会访问网页。直接启动 `server.py` 仍会加载原应用的真实依赖，
不是本次离线入口。

## 运行（Python 3.10+，核心和演示无第三方依赖）

在仓库根目录：

```powershell
python -B -m agent.infrastructure.research_demo --scenario conflict
python -B -m agent.infrastructure.research_demo --scenario knowledge
python -B -m agent.infrastructure.research_demo --scenario missing
python -B -m agent.infrastructure.research_demo --scenario conflict --max-tool-calls 2
python -B -m unittest discover -s tests -p test_research_loop.py -v
```

四场景预期：`completed / completed / needs_input / budget_exhausted`。
CLI 输出 simulation 标记、终态、缺口、动作/观察 trace 和预算计数。
knowledge 只调用知识库；conflict 根据观察补查正文及统计口径；missing 无资料则
求助，说明框架没有强制每次走满全部工具。

可选真实 LangGraph 图集成测试（建议独立虚拟环境）：

```powershell
python -m pip install -r tests/requirements-graph.txt
python -B -m unittest discover -s tests -p 'test_research*.py' -v
```

执行真实 Editor 和 Supervisor 图，只替换外部模型、DB、写作等依赖，并禁止网络
连接。未安装可选依赖时图测试明确 SKIPPED，不冒充通过。

## 决策协议与结果反馈

`DecisionModel.decide(request, max_output_tokens=...)` 接收任务、三工具 schema、
已接受的动作/观察历史、剩余预算。一次返回一个 tool/finish/need_input 动作。

```json
{
  "action": "tool",
  "tool": "search_web",
  "arguments": {"query": "主题 2026 官方统计口径"},
  "reason": "现有资料过旧，需要补充公开来源",
  "gaps": ["缺少当前年度数据"],
  "conflicts": []
}
```

工具执行后，下一轮 history 包含该动作及 `call_id/status/evidence/error` 观察。
证据有 `id/kind/title/text/url/published_at/truncated`；ID 由运行时分配，后端
不能伪造。search_web 为 snippet，fetch_page 为 page，知识库为 knowledge。
空结果、执行拒绝和错误同样反馈模型。工具材料是不可信数据，不能改变执行权限。

每轮模型更新 gaps/conflicts。finish 必须包含 answer/citations，引用必须真实
存在，至少包含已读取的正文或知识片段，且未声明剩余缺口/冲突；否则输出
needs_input。reason 只保存简短操作说明，不记录隐式思维链。

**这些是结构性门禁，不是真实性保证。** 资料是否最新、是否相互矛盾、每句话是否
被引用支持，仍需未来模型与独立评估。本框架不能阻止模型错误清空语义缺口。
completed 也不等于已经通过外层 Reviewer。真实接入前需补充时效性、矛盾、无证据
和提示注入评测。

## 工具、预算与终态

- `search_web(query)`：最多 5 条公开摘要，包含标题、URL、可选发布日期。
- `search_knowledge_base(query)`：返回原始知识片段，不调用带 LLM 总结的旧工具。
- `fetch_page(url)`：只读取本轮搜索已发现且通过公开 URL 语法检查的地址。

只允许对应 query/url 参数，不允许模型传 user_id、路径、命令或任意工具名称。
未来后端必须由宿主绑定认证用户与知识库 ACL。URL 检查仅是语法限制，**不是生产
SSRF 防护**；实际 HTTP 适配器须补充 DNS/每次重定向后的公网检查、网络出口限制、
响应大小上限、超时与脱敏，不能未经审查直接复用旧 WebScraper 当作安全沙箱。

默认 ResearchLimits：12 次决策、8 次工具执行、总时长 60 秒、单次超时 10 秒、
80000 token 估算单位、单次输出 2048 估算单位。预算属于**一次章节循环**，不是
整份报告共享上限。非法/重复动作不执行但消耗决策步数；失败/超时的实际调用计数。

Token 按请求/输出 JSON 的 UTF-8 字节数保守估算，重复输入的历史也计入；调用前
预留输出预算。它不是账单 Token，也不声称适配所有 tokenizer。真实 provider
必须限制输出，并使用 tokenizer/usage 校准。超时与取消依赖协作式异步适配器，
不能强杀任意阻塞进程。可用 asyncio.Event 取消直接运行；宿主全报告取消 token
接线和共享预算未包含在本次改造中。

终态：`completed / needs_input / budget_exhausted / failed / cancelled`。
非完成态不提供正式 answer，保留证据、trace 和缺口。相同工具/参数重复调用被拒绝，
模型需改查询或退出。needs_input 是待人工处理结果，不是已实现的审批 UI。

## 现有项目接入

```python
from agent.research_agent import ResearchAgent
from agent.infrastructure.research_loop import ResearchLoop, ResearchLimits
from agent.infrastructure.research_demo import DemoBackend, DemoDecisionModel
from agent.tools.research_tools import ResearchTools

researcher = ResearchAgent(research_loop=ResearchLoop(
    DemoDecisionModel(), ResearchTools(DemoBackend()), ResearchLimits()))
result = await researcher.run_depth_research("查询内部制度")
```

未来模型只实现 DecisionModel.decide：把 schema/history 转为厂商的工具调用消息，
把返回结果转成上述动作。数据后端只实现 ResearchBackend 的三个异步方法，不用
修改循环。**禁止将 DemoDecisionModel 用于真实研究。**

应用依赖准备好后，在启动组合入口创建 `EditorAgent(researcher=researcher)`，
把该实例提供给 Supervisor 的 editor。不要在正在运行的全局 Agent 上切换用户或
模型。EditorAgent 仍有旧应用导入依赖，不是纯离线入口；本次只为 ResearchAgent
和循环去掉了急切模型初始化。初步调研/规划/写作模型不属于这次三工具改造。

ResearchAgent 保留 section/draft/sources/revisions，新增 research_status 和
research_result。Editor 保留终态与证据，未完成章节直接结束子图；Supervisor
仅在全部预期章节完成后进入 writer/publisher，否则显示缺口并结束。直接调用
writer/publisher 也有检查。字符串流协议不变，“流结束”不代表“研究完成”。

旧流水线记录为 legacy，保持兼容，不声称为旧路径补齐同等证据门禁。
未实现：工具循环逐步 checkpoint、审批恢复、真实 provider、全报告成本计量、
生产部署。needs_input 不会进入原本默认 approve 的 HumanAgent。
