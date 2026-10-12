# Smart Agent

<p align="center">
  <strong>共享 Agent 内核 · Code Agent · Career Agent</strong>
</p>

<p align="center">
  <img alt="Python" src="https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white">
  <img alt="Tests" src="https://img.shields.io/badge/tests-745%20passed-blue?style=flat-square">
  <img alt="License" src="https://img.shields.io/badge/license-MIT-blue?style=flat-square">
</p>

Smart Agent 是一个面向多领域任务的可扩展大语言模型 Agent。共享内核负责模型调用、**单一 ReAct 循环**、工具、Skill、Session、记忆、上下文压缩和权限；当前提供两个 Domain：

| Domain | 主要职责 | 当前状态 |
|:--|:--|:--|
| Code Agent | 仓库分析、编码、调试、测试和代码审查 | 已实现 |
| Career Agent | 个人材料库、JD 分析、证据化简历、独立核验、自适应面试和网申准备 | 离线核心闭环已实现；真实 API 待验收 |

首版由用户手动选择 `code` 或 `career`。Domain 与 Session 固定绑定，切换 Domain 需要创建新 Session，暂不使用自动分类模型。

> 当前边界：仓库测试为 `745 passed, 3 skipped`。真实 Career 生成、独立 Resume Verification、Embedding、Reranker、OCR、豆包 ASR/TTS 和真实招聘网站仍需要外部配置和单独验收，Mock 与合成数据不代表真实供应商效果。

## 项目定位

Career Agent 主要解决研究生求职中的两个高频问题：

1. 将长期个人材料可靠地组织为针对单个 JD 的专属简历。
2. 使用 JD、核验简历、用户专业知识库和回答反馈进行可追溯模拟面试。

系统不把大模型生成内容直接当作用户事实。个人材料先形成候选事实，只有用户确认后的证据才能进入简历；最终简历还要交给不同模型、不同 API 通道的 Resume Verification Agent 核验。

项目的核心研究命题是：

> 由 JD、核验简历、专业知识库和回答反馈联合驱动的证据约束、可追溯、自适应求职 Agent。

RAG、多 Agent、ChromaDB、CP-SAT、Playwright、语音和 PDF 是实现手段，不单独作为创新点。

## 总体架构

```text
用户手动选择 Domain
        |
        v
可选 Root Supervisor
Goal -> Plan -> Dispatch -> Execute -> Verify -> 条件 Re-plan
        |
        +-------------------+
        |                   |
        v                   v
   Code Agent           Career Agent
 single ReAct loop    single ReAct loop
        |                   |
        +---------+---------+
                  v
          Shared Smart Agent Core
  model / tools / skills / session / memory /
  context compaction / permissions / audit
```

> 注：Root Supervisor 的 `Plan / Dispatch / Verify` 属于**外层编排**，作用于跨步骤目标；
> 内核的单一 ReAct 循环作用于**单轮内的模型决策**。两者不是同一层——Supervisor 不会把单轮拆成
> `explore / execute / verify` 阶段，单轮始终由模型自决。

### 架构原则

- **手动 Domain 选择**：首版避免分类误路由，后续增加模块时再评估自动路由。
- **Session 固定 Domain**：恢复 Session 时沿用原 Domain，避免运行中替换系统能力。
- **轻量 Root Supervisor**：只负责目标准入、规划、委派、结构化验证和条件重规划；简单任务直接进入 Domain Agent，不调用根模型。
- **渐进能力披露**：实际能力为 `Core Tool/Skill + Domain Tool/Skill + Stage Tool/Skill`。
- **最小 Artifact 交接**：子 Agent 只接收完成任务所需的数据，不继承完整父会话。
- **可插拔实现**：检索、规划、核验、面试、语音和浏览器均保留接口注入点，支持消融实验。

## Career Agent

### 完整数据流

```text
长期个人材料库
  -> CandidateFact 候选事实
  -> 用户确认
  -> PersonalEvidence
  -> 新 JD 创建 ResumeRun
  -> JDRequirement 逐条匹配
  -> RequirementMatchCache 复用或重新检索
  -> EvidenceMatch
  -> ResumePlan
  -> ResumeDocument
  -> 独立 Resume Verification
  -> PDF + trace.json
  -> 模拟面试 / 网申准备
```

个人材料导入与单次简历生成互相独立。材料可以随时新增、修订和删除；创建新的求职任务时只提交 JD 和必要的个人基本信息，不重复上传全部材料。

### 1. 长期个人材料库

支持六类材料：

- 通用简历参考
- 教育经历
- 实习经历
- 项目/作品经历
- 获奖经历
- 科研成果

原始文件、解析文本、内容哈希、解析器版本和修订号会持久化保存。规则或模型抽取内容先进入 `CandidateFact`，只有用户确认后才能成为 `PersonalEvidence`。

通用简历只是只读参考源，不是事实真值；模拟面试使用当前求职任务中新生成并核验通过的简历。

### 2. 项目与作品入库

- 项目库支持工作区路径和 ZIP；真实项目与个人项目经历分开保存。
- `Project Analyst Agent` 只读查看代码、README、配置、PDF、DOCX、PPTX、XLSX 和图片，不执行项目代码。
- 环境、依赖、数据集、模型权重、缓存和密钥不会进入个人材料库或向量库。
- 每个项目只生成一个带版本和来源定位的 `ProjectDescriptionDocument` 候选档案。
- 个人职责、历史成果和指标必须经过用户确认，确认后才生成 `ProjectEvidence`。
- JD 先匹配六大个人材料板块，只有项目档案出现相关信号且超过阈值时才建议深入项目库。

### 3. JD 与 ResumeRun

每次提交 JD 会创建独立 `ResumeRun`，并关联：

- `JDDocument` 和多个 `JDRequirement`
- `EvidenceMatch`
- `ResumePlan`
- `ResumeDocument`
- `VerificationReport`
- PDF Artifact

JD 支持文本输入和图片 OCR 候选文本。OCR 结果必须由用户校对后再进入 JD 分析。

ResumeRun 支持历史恢复和软删除。删除求职会话后，界面不再展示该任务，也不能继续生成、核验或导出，但 JD、要求、证据匹配和要求级缓存仍保留。

### 4. 要求级匹配缓存

`RequirementMatchCache` 的最小复用单位是单条 JD 要求，不是整份简历：

- 检查语义、关键词、类别、岗位族、职级和必选属性。
- 检查材料、事实、检索后端、Embedding 和配置版本。
- 相关材料变化时只让依赖缓存过期。
- 新增无关类别材料不会使全部缓存失效。
- 复用结果保留来源、依赖、版本和 `reused_from` 信息。

这使相似岗位要求可以减少重复检索，同时避免材料库变化后继续使用过期证据。

### 5. 约束驱动 ResumePlan

ResumePlan 不是固定简历模板。系统根据用户实际拥有的教育、实习、项目、获奖和科研内容动态选择章节，没有的部分不会生成空章节。

默认规划器使用 OR-Tools CP-SAT，同时考虑：

- JD 要求覆盖
- 事实证据支持
- 页面与字数预算
- 内容重复组
- 岗位 TITLE 和相关度
- 经历背书与信息价值

规划器、检索器和构建器均可替换为实验变体。

### 6. 独立简历核验与 PDF

简历生成后由 Web 前端自动调用 Resume Verification，不要求用户额外点击“独立核验”。

- 核验 Agent 使用独立 Session、零 Tool 权限和最小 Artifact。
- 运行时要求生成模型与核验模型名称不同、API 端点不同。
- 核验器只返回结构化问题，不直接修改个人事实。
- 修订器只能修改核验报告点名的 Claim，证据和 JD 引用保持冻结。
- PDF 使用语义化 HTML/CSS 和 Playwright Chromium 渲染。
- 导出时检查页数、文本层、字体、横向溢出和 Claim 完整性。
- 每次同时保存 HTML、PDF 和 `.trace.json`，追溯到事实、证据、JD 要求和原始材料。

离线闭环和约束测试已经完成；真实异模异 API 核验仍需供应商密钥验收。

### 7. 自适应 Interviewer Agent

Interviewer Agent 拥有独立 `InterviewSession`，Root/Career 不参与逐轮问答。

#### 真实模拟面试

- 输入为目标 JD、核验简历和面试前冻结的知识包。
- 热路径不调用 DeepSearch 或实时外部检索，降低延迟和不确定性。
- 禁止读取原始个人材料、项目仓库、Career 主会话和其他求职任务。
- 系统根据 JD、简历和随机种子分配面试官，用户不能手工选择现实中的面试官。

当前真实模拟画像包括：

- 均衡资深面试官
- 深挖领域专家
- 广度招聘面试官

#### 查漏补缺面试

- 不强制要求真实 JD 或简历。
- 可以逐题实时读取用户授权的专业知识 RAG。
- 优先询问长期掌握度中较薄弱的主题。
- 结束后按 `interview_id` 幂等更新知识掌握状态。

#### 自适应策略

- 人物画像的压力、广度偏好、职级和性格参与问题决策，不只是改变说话语气。
- 问题效用记录 JD 相关性、简历相关性、知识不确定性、薄弱程度、画像偏好、重复度和追问成本。
- 下一动作限定为 `ASK / FOLLOW_UP / SWITCH_TOPIC / END`。
- 默认单主题最多追问三次，并记录父问题、追问深度和停止原因。
- Bayesian belief 只估计专业知识掌握，不判断项目真假。
- 复盘同时输出面试官主观结论和 Career 学习建议，不输出录用预测。

当前缺口：`InterviewSessionState` 已固定 JD 和简历的 ID/revision，但尚未显式保存 `resume_run_id`；Web 启动真实面试时仍存在回退到最新 JD/简历的兼容行为，下一步需要移除该行为。

### 8. 面试知识库

- 支持 TXT、Markdown、PDF 和 DOCX。
- 保存租户、专业、子专业、主题、难度、权威等级、章节、页码、内容哈希和来源位置。
- 一份资料可以有多个专业和主题标签。
- 根据 JD 构建岗位能力树，再执行元数据过滤。
- 支持 BM25、向量、混合召回和 Reranker。
- 用户资料按不可信输入处理，不能覆盖系统提示词或 Tool 权限。
- 无可靠来源时明确降级，不把模型内部知识伪装为可追溯资料。
- 面试知识与 PersonalEvidence 完全隔离。

### 9. 语音面试边界

当前已实现供应商中立的 ASR/TTS Adapter、Partial/Final 事件、播放状态、打断取消和录音同意边界。语音层不拥有 Interview Session、RAG、策略或记忆。

完整低延迟链路尚未完成，目标架构为：

```text
浏览器麦克风音频帧
├── VAD：开口、停顿、打断
└── 流式 ASR：持续识别
        ↓
Turn Endpoint Manager
        ↓
UserTurnFinal
        ↓
Interviewer Agent
        ↓
流式 LLM -> 流式 TTS
```

待完成内容包括浏览器麦克风实时流、VAD、`SpeechSegmentFinal`/`UserTurnFinal`、流式 WebSocket、豆包真实协议和延迟实测。因此当前不能宣称“低延迟语音面试已完成”。

### 10. Application Agent

Application Agent 是半自动网申填报助手，不是全自动投递系统：

- 只接收 URL、核验简历和用户确认字段。
- Playwright Adapter 支持 HTTPS 域名白名单、页面快照、字段扫描、填写和上传边界。
- 字段分为 `AUTO_FILLED / NEEDS_CONFIRMATION / MISSING / SENSITIVE`。
- 只有核验简历中有直接证据的字段可以自动填写。
- 缺失、歧义、开放题和敏感字段保持空白。
- 不绕过验证码、登录保护或反自动化机制。
- 最终提交、撤回、发信和修改已提交申请没有自动执行接口。

当前已有协议、Playwright Adapter、本地模拟表单测试和字段规划 API；真实招聘网站产品闭环尚未验收。

## Code Agent

Code Agent 保留原有编码能力：

- 仓库分析、文件搜索、读写、编辑和命令执行。
- 测试运行、文件检查点和回退。
- 正则/AST 宽松审查。
- 严格 Code Review 子 Agent。
- Import Map 和 Docker 沙箱测试。
- 审查发现向 Skill/记忆沉淀。

Code Review Agent 只属于 Code Domain。Career 的核验是 Resume Verification Agent，不执行代码扫描。

Code 审查模式：

```bash
SMART_AGENT_REVIEW_MODE=off      # 不执行代码审查
SMART_AGENT_REVIEW_MODE=loose    # 正则和 AST 宽松审查
SMART_AGENT_REVIEW_MODE=strict   # 宽松审查 + Code Review 子 Agent
```

## 共享内核

### Agent 循环

执行循环是**单一 ReAct 循环**：模型每步自决"回答 / 追问 / 调用工具"，工具观察追加后再次决策；
不使用 `explore / execute / verify` 阶段去门控模型（旧阶段标签仅为遥测兼容保留，见
`smart_agent/turn_kernel.py`）。约束与兜底下沉到四处：

- **任务契约**：按意图与实际副作用在 `conversation / read_only / mutation / execution` 之间升级，越权工具直接被拒。
- **运行时状态机**：提供硬门控，不依赖提示词约束。
- **有界 self-reflection**：仅工具失败时触发一次，结果 advisory，不能执行工具、不能绕过任务契约与核验契约。
- **失败可恢复**：一次失败后重试成功不算污染整轮，避免"一次失败就全盘重来"。

```text
Prelude
  -> 意图、上下文、权限和审查初始化
Recurrent Kernel（单一 ReAct 循环）
  -> 模型决策：回答 / 追问 / 调用工具
  -> 工具观察追加 -> 再次决策
  -> 失败时触发一次有界 self-reflection（advisory）
Coda
  -> 审查发现、记忆、Skill 和 Session 收尾
```

### 四级上下文压缩

上下文压缩属于共享 Core，不按 Domain 分叉：

- 达到上下文窗口 85% 时触发。
- 同一次压缩按需执行 L1 至 L4，降到约 55% 后停止。
- 优先保留系统约束、用户目标、强制内容和最近 Turn。
- 压缩前将完整上下文归档到当前 Session。
- 后续重压缩从归档原文生成，避免反复摘要导致信息漂移。

### Tool 与 Skill

- Core 提供通用文件、命令、记忆和 Session 能力。
- Code/Career 分别注册专属 Tool 与 Skill。
- Career 再按 `materials / resume / interview / application` Stage 渐进披露。
- 运行时状态机提供硬门控，Agent 负责理解目标和选择下一动作。

## 当前状态

| 能力 | 状态 |
|:--|:--|
| Domain Runtime、Session 固定 Domain | 已完成 |
| Root Supervisor | 已实现；支持 `off/auto/force`，旧 P-E-RP 配置兼容 |
| Code Agent、Code Review Agent | 已完成 |
| 长期个人材料库与事实确认 | 已完成 |
| 项目库、Project Analyst 与项目档案确认 | 已完成；真实多行业项目评测待补 |
| JD、ResumeRun、要求级缓存 | 已完成 |
| ResumePlan、简历、独立核验、PDF | 离线闭环完成；真实异模异 API 待验收 |
| 文字 Interviewer Agent | 已完成双模式内核 |
| 真实面试显式绑定 ResumeRun | 未完成 |
| 面试知识库与混合检索 | 已完成；真实模型指标待验收 |
| Career Web 工作台 | 已完成首版 |
| OCR | Adapter 和流程完成；真实准确率待测 |
| 语音面试 | Adapter 和协议基础完成；流式热路径未完成 |
| Application Agent | 协议和 Playwright Adapter 完成；真实网站未验收 |
| 离线 E2E 与消融 | 已完成 |
| 真实 Career E2E | 未完成 |

## 快速开始

### 1. 安装

最低运行环境为 Python 3.10+。

```bash
pip install -e .
```

Web、Career、OCR 和开发依赖：

```bash
pip install -e ".[dev,web,career,ocr]"
python -m playwright install chromium
```

### 2. 配置

所有配置示例见 `.env.example`，只填写实际使用的供应商，不要提交真实密钥。

通用 OpenAI 兼容模型：

```bash
CUSTOM_API_BASE_URL=https://your-api-endpoint/v1
CUSTOM_API_KEY=your-api-key
CUSTOM_MODEL=your-model
```

Career 生成模型可以使用豆包 Ark：

```bash
CAREER_GENERATION_PROVIDER=doubao
CAREER_GENERATION_BASE_URL=https://ark.cn-beijing.volces.com/api/v3
CAREER_GENERATION_API_KEY=your-ark-api-key
CAREER_GENERATION_MODEL=your-doubao-endpoint-id
```

Resume Verification 必须使用不同模型和不同 API 端点：

```bash
CAREER_VERIFIER_PROVIDER=openai
CAREER_VERIFIER_BASE_URL=https://api.openai.com
CAREER_VERIFIER_API_KEY=your-independent-key
CAREER_VERIFIER_MODEL=gpt-4o-mini
```

事实抽取、JD Analyzer、面试问题生成、面试回答评价、Embedding 和 Reranker 均支持独立配置，变量名见 `.env.example`。

### 3. 启动 Web

```bash
smart-agent-web
```

或：

```bash
python -m smart_agent.web.app
```

访问 `http://127.0.0.1:8341/`。

Web 工作台提供：

- Code/Career 手动切换。
- Career 长期个人材料库。
- 新建空白求职任务。
- 最近 3 条会话和完整历史会话。
- 会话搜索、恢复和删除。
- JD、证据、简历、核验、PDF、文字面试和网申计划。

### 4. CLI 与 Headless

```bash
smart-agent --domain code
smart-agent --domain career

smart-agent-headless --domain career "根据这个 JD 分析岗位要求"
smart-agent-headless --supervise --domain career "根据这个 JD 生成并核验简历"
```

`--supervise` 强制启用 Root Supervisor；`--orchestrate` 仍作为兼容别名。配置
`SMART_AGENT_SUPERVISOR=auto` 时，简单任务自动旁路，复杂任务才进入外层。

### 5. 离线 Career 演示

不配置外部 API 也可以使用合成数据跑通离线闭环：

```bash
python scripts/career_e2e.py --workspace .tmp/career-e2e --force
```

该脚本会导入材料、确认候选事实、分析 JD、检索证据、规划简历、执行离线核验、渲染 PDF，并完成文字面试。它使用启发式抽取、离线向量、CP-SAT 和离线核验器，不代表真实模型质量。

### 6. 真实 API 测试

配置所有必需供应商后显式启用：

```bash
CAREER_LIVE_API_TEST=1 python -m pytest tests/test_career_live_api.py -q
```

真实验收必须保存模型/API 身份、延迟、Token、成本和失败样本。

## 测试与评测

完整测试：

```bash
python -m pytest tests -q
```

当前实测结果：

```text
745 passed, 3 skipped
```

文档链接：

```bash
python scripts/check_docs.py
```

离线消融：

```bash
python scripts/run_ablations.py --output docs/evaluation/results/ablation-v1
```

固定数据、随机种子和原始结果位于：

- `fixtures/ablation/competition_v1.json`
- `docs/evaluation/ablation-protocol.md`
- `docs/evaluation/results/ablation-v1/`

当前离线实验只验证协议和错误边界，不能外推到真实岗位、专业和用户。

### 已知静态检查债务

```bash
python -m ruff check .
```

当前全仓 Ruff 有 103 个既有问题，主要位于旧 benchmark、维护脚本和历史测试。CI 配置了全仓 Ruff，因此在这些问题修复或检查范围被合理划分前，不能宣称 CI 全绿。

## Web API

主要 Career 接口：

| 接口 | 用途 |
|:--|:--|
| `GET /api/career/workspace` | Career Artifact 和阶段状态 |
| `GET /api/career/material-library` | 长期个人材料库 |
| `POST /api/career/materials` | 分类导入个人材料 |
| `DELETE /api/career/materials/{material_id}` | 使材料失效，保留历史快照 |
| `POST /api/career/projects/import` | 导入项目/仓库 |
| `POST /api/career/projects/{project_id}/description/generate` | 生成项目描述候选稿 |
| `POST /api/career/projects/{project_id}/description/confirm` | 确认项目描述 |
| `POST /api/career/facts/decision` | 确认或拒绝候选事实 |
| `POST /api/career/ocr` | OCR 候选文本 |
| `POST /api/career/jd/analyze` | 创建 ResumeRun 并拆分 JD |
| `POST /api/career/requirements/match` | 要求级缓存与证据匹配 |
| `POST /api/career/resume/build` | 生成 ResumePlan 和简历 |
| `POST /api/career/resume/verify` | 独立核验 |
| `POST /api/career/resume/render` | 生成 PDF 和追踪侧车 |
| `GET/POST /api/career/interview-knowledge` | 面试知识库 |
| `POST /api/career/interviews/start` | 创建文字面试 |
| `POST /api/career/interviews/{id}/answer` | 提交回答 |
| `DELETE /api/career/sessions/{type}/{id}` | 删除求职或面试会话 |
| `POST /api/career/application/plan` | 生成网申字段计划，不提交 |

## 项目结构

```text
smart-agent/
├── smart_agent/
│   ├── agents/              Root、Code Review、Career 子 Agent
│   ├── builtin_skills/      随包发布的 Career Skill
│   ├── career/              Career 数据、检索、简历、面试、语音、网申
│   ├── domains/             Domain 注册、Session 绑定、能力装配
│   ├── evaluation/          消融实验实现
│   ├── review/              Code Review 系统
│   ├── tools/               Core、Code、Career Tool
│   ├── tui/                 TUI
│   └── web/                 FastAPI/WebSocket 后端
├── frontend/                当前 Web 前端
├── tests/                   自动化测试
├── fixtures/                合成演示与固定评测数据
├── benchmark/               Code 审查和编码基准
├── scripts/                 E2E、消融、截图和维护脚本
├── docs/                    架构、设计、评测和参赛文档
├── .env.example             配置模板
└── pyproject.toml           项目依赖和入口
```

## 重要文档

- `项目总结.md`：当前完整交接说明。
- `后续工作.md`：材料库、ResumeRun 和要求级缓存的原始需求。
- `docs/competition/smart-agent-rebuild-roadmap.md`：总体实施路线。
- `docs/competition/smart-agent-remaining-checklist.md`：实际验收清单。
- `docs/competition/smart-agent-competition-report.md`：参赛报告草稿与引用。
- `docs/competition/career-demo-flow.md`：演示流程。
- `docs/evaluation/ablation-protocol.md`：消融实验协议。

## 下一步

1. 修复或合理划分当前 Ruff 静态检查债务。
2. 让真实模拟面试显式绑定 `resume_run_id`，删除“最新 JD/简历”回退。
3. 使用真实生成 API 和独立核验 API 跑通 Career 全链路。
4. 完成浏览器流式语音、VAD/Endpoint Manager 和豆包 ASR/TTS 验收。
5. 使用真实脱敏样本补充实验、延迟、成本、失败案例和比赛报告。
6. 最后拆分当前单文件前端，不提前增加无关 Domain 或子 Agent。

## License

MIT
