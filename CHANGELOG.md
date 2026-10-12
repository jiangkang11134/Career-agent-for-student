# Changelog

## 0.3.0 (2026-08-07)

### 统一
- 项目、Python 包、CLI、环境变量和运行目录统一�?Smart Agent�?- 后台包迁移到 `smart_agent/`，Web 入口迁移�?`smart_agent/web/app.py`�?- 只保�?`smart-agent`、`smart-agent-headless` �?`smart-agent-web` 三个命令�?- 内置 Career Skill 收入 `smart_agent/builtin_skills/` 并随包发布�?
### 清理
- 移除旧品牌目录、临时缓存和源码目录内的本地设置文件�?- 用户配置、会话、记忆和 Career Artifact 统一存放�?`.runtime/`�?
## 0.2.0 (2026-07-10)

### 新增
- 两级代码审查系统（宽�?0.7ms + 严格 ~5s�?- Docker 沙箱测试（sandbox_test 工具�?- 运行时审查模式切换（off/loose/strict�?- 三级记忆系统（BM25 六维评分搜索�?- 成本上限控制（SMART_AGENT_API_COST_LIMIT�?- /report 运行报告命令
- CI/CD 流水线（GitHub Actions�?
### 改进
- 核心循环�?1225 行单体拆�?3 个模�?- review agent �?5 �?LLM 调用降为 1 �?- test agent �?LLM 调用直接�?sandbox_test
- 全部 103 文件零编译错�?- 全量 Ruff lint + mypy 配置

### 修复
- 权限系统非交互模�?auto-allow
- 白名单行级前缀检�?- cost_tracker Decimal/float 类型错误
- 200+ 处注释残留导致变量未定义
- Windows GBK 编码问题
