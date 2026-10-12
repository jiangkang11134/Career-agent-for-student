---
triggers: [个人材料, 简历导入, 实习经历, 项目经历, 候选事实, 事实确认]
agent_types: [career]
stages: [materials, resume]
priority: 90
---

# Career Material Evidence

将用户材料转换为可追溯、可确认的个人证据，同时禁止把抽取结果直接当成事实。

## 执行方法

1. 多步骤任务先调用 inspect_career_workspace，根据当前 Artifact 缺口自行选择下一步。
2. 按 resume、education、award、internship、project、research 分类导入材料。
3. 通用简历只作为只读 ResumeSource，不能直接生成已确认事实。
4. 使用 extract_career_facts 生成候选事实；候选事实不能支持简历陈述。
5. 向用户集中展示候选事实，由用户决定导入或拒绝。
6. import_candidate_facts 只创建 UNVERIFIED PersonalEvidence。
7. 只有用户明确确认后才能调用 confirm_career_facts。
8. 项目检索结果只是材料片段，不能替代 PersonalEvidence。

## 禁止事项

- 不推断用户没有提供的公司、职位、日期、指标和技术掌握程度。
- 不把通用简历、项目 RAG 命中或面试知识当作已确认个人事实。
- 不为了继续流程而自动确认候选事实。
