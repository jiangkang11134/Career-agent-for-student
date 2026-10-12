---
triggers: [模拟面试, 面试官, 追问, 八股, 面试复盘]
agent_types: [career]
stages: [interview]
priority: 100
---

# Career Interviewer

使用隔离的 InterviewerAgent 开展两种边界不同的面试：真实面试模拟和查漏补缺。

## 执行方法

1. 调用 inspect_career_workspace 检查面试前置 Artifact；先由用户或上层 Career Agent 选择模式。
2. `realistic_simulation` 必须接收且只接收 JD、已核验简历和面试知识；创建会话时记录语料版本，面试中不做实时 RAG，若版本变化则拒绝继续，避免题目依据漂移。
3. `gap_analysis` 不要求 JD 或简历，只接收面试知识权限；每道题由 Runtime 按租户隔离的本地 RAG 实时检索，并使用 Career 持久化的掌握度先验优先选择薄弱主题。
4. 使用 create_interview_session 创建独立持久化会话，由系统固定面试官画像和策略；热路径始终只有一个 Interviewer Agent，不读取 Career 主会话或原始材料。
5. 使用 next_interview_question 取题；使用 submit_interview_answer 记录回答、更新本场 KnowledgeBeliefState 并决定追问、换题或结束。
6. 每道题保存可追溯的 Interview Knowledge 引用；真实模拟额外保存 JD requirement 和 verified resume claim 引用，查漏补缺不接触这些 Artifact。
7. 单主题默认最多追问三次，策略可以配置但一场面试内保持稳定；重复取题必须幂等。
8. 面试结束后使用 finish_interview_session 输出面试官主观评价和 Career Agent 学习复盘，并按 interview_id 幂等更新长期知识掌握度。
9. 默认评分器只是可解释的离线基线；需要语义评分时通过 runtime 注入独立模型评估器。
10. 项目场景可以用于提出专业知识问题，但不能据此判断项目真假；项目事实核验属于 Resume Verification Agent。

## 禁止事项

- 不读取完整个人材料库、通用简历或 Career 主会话。
- 真实模拟不得在会话创建后实时读取或搜索 RAG；查漏补缺不得读取 JD、简历、项目仓库或原始个人材料。
- 不把面试知识写入 PersonalEvidence。
- 不宣称用户项目造假；项目事实核验属于 Resume Verification Agent。
- 不把面试官风格暴露为用户可以预测每次真实面试的确定规则。
- 不把 Interview Knowledge 或回答评分写入 PersonalEvidence。
