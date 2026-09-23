# 工程回归台账

## DEEPRESEARCH-TOOL-LOOP-20260923

- 范围：ResearchAgent 可选三工具循环及 Editor/Supervisor 终态传播。
- 保护条件：白名单仅三工具，模型参数不能控制用户作用域；新循环没有证据或
  未完成时不得被 reviewer/writer/publisher 转成成功。保留旧流水线和字符串流。
- 可复现缺陷（INTRODUCED，已修复）：模型返回 JSON action=[]/{} 时集合判断
  抛出 TypeError。修复为先验证字符串类型，回归要求 failed/invalid_model_output。
- Windows 短超时回归（INTRODUCED，已修复）：事件循环定时器与低精度 monotonic
  时钟交错，10ms 总预算偶发被记录为 step_budget。改用高精度 perf_counter，并在
  等待超时时按实际限制等待的总预算/单调用预算分类，不降低测试断言。
- 验证：test_research_loop.py 覆盖缺证据、伪造引用、预算/超时/取消、非法参数和
  输出、观察反馈及 legacy 兼容；test_research_graph_integration.py 使用真实
  LangGraph 验证未完成章节停止子图和主图，禁止写作/发布/知识库存储。
- Oracle：固定合成资料及预期工具序列；知识足够必须跳过搜索；缺资料必须求助；
  任一章节不完整时写作与发布调用计数必须为 0。独立只读复核复现并复验畸形输出。
- 边界：模拟策略不是模型准确率评测；URL 语法检查不替代 DNS/重定向防护；
  token 为估算，预算按章节；真实数据适配器及审批恢复不在本次范围。
