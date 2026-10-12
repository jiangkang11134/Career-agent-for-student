---
triggers: [岗位简历, JD分析, 生成简历, 简历核验, 简历PDF]
agent_types: [career]
stages: [resume]
priority: 100
---

# Career Resume Workflow

根据用户目标自主编排简历任务，并以证据门控保证事实准确。

## 执行方法

1. 调用 inspect_career_workspace 获取 Artifact Index、推荐阶段和硬门控。
2. Agent 自己判断下一动作，不要求用户机械执行固定步骤。
3. 将 JD 分解为逐条 JDRequirement，保留原文位置。
4. 只检索已核验或用户确认的 PersonalEvidence；无匹配时保留 Evidence Gap。
5. 使用 ResumePlan 选择内容，兼顾 JD、事实强度、公司与 Title 背书、成果和篇幅。
6. 从 ResumePlan 生成结构化 ResumeDocument，不能直接生成 PDF。
7. 使用独立模型和独立上下文核验简历。
8. 只修改核验报告点名的 Claim，最多执行两轮自动修订。
9. 只有核验通过后才能调用 render_resume_pdf。

## 强制边界

- 不得引用 UNVERIFIED、NEEDS_REVERIFICATION 或 INVALIDATED 事实。
- 不得为了覆盖 JD 而编造经历或技术点。
- 不得让生成模型替代独立核验模型。
- 不得将缺口静默改写为用户能力。
