"""Smart Agent Headless Runner --- 非交互式一次性执行。 灵感来源于 Hermes Agent 的 headless 模式，适用于 CI/CD 流水线和
自动化工作流。

用法:
  # 运行单个提示并退出
  python -m smart_agent.headless "帮我分析这个项目的结构"

  # 管道输入
  echo "解释这段代码" | python -m smart_agent.headless

  # 在 Docker 中
  docker compose run --rm headless "修复这个 bug"
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import uuid
from pathlib import Path


def _write_headless_messages_trace(
    trace_path: str | None,
    *,
    cwd: str,
    prompt: str,
    runtime: dict | None,
    result_messages: list[dict] | None,
    response_text: str | None,
    error_text: str | None = None,
) -> None:
    """将对话消息追踪写入 JSON 文件。

    参数:
        trace_path: 输出文件路径。若为 None 或空字符串则跳过写入。
        cwd: 当前工作目录。
        prompt: 用户输入的提示词。
        runtime: 运行时配置字典。
        result_messages: 对话消息列表。
        response_text: 助手的响应文本。
        error_text: 错误信息文本（可选）。

    返回:
        None
    """
    if not trace_path:
        return
    payload = {
        "cwd": cwd,
        "prompt": prompt,
        "model": (runtime or {}).get("model"),
        "messages": result_messages or [],
        "assistant_response": response_text,
        "error": error_text,
    }
    path = Path(trace_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )


def _allow_edits_requested(cli_flag: bool = False) -> bool:
    """判断 headless 模式是否应自动批准编辑、命令和超出工作目录的访问。

    选择加入的非交互式 CI 模式。通过 --allow-edits 标志或
    SMART_AGENT_ALLOW_EDITS 环境变量（1/true/yes/on）控制。

    参数:
        cli_flag: 命令行标志是否设置。

    返回:
        如果应自动批准编辑，返回 True；否则返回 False。
    """
    if cli_flag:
        return True
    return os.getenv("SMART_AGENT_ALLOW_EDITS", "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def _make_auto_approve_prompt():
    """构建一个非交互式权限提示处理器，自动批准当前运行的所有请求。

    决定是会话范围的（不持久化到权限存储）：编辑通过 allow_all_turn，
    路径/命令通过 allow_once。

    返回:
        自动批准处理函数。
    """

    def _auto_approve(request: dict) -> dict:
        """自动批准单个权限请求。

        参数:
            request: 权限请求字典，包含请求的 kind 等信息。

        返回:
            包含批准决策的字典。编辑请求返回 allow_all_turn，
            其他请求返回 allow_once。
        """
        if request.get("kind") == "edit":
            return {"decision": "allow_all_turn"}
        return {"decision": "allow_once"}

    return _auto_approve


def run_headless(
    prompt: str | None = None,
    allow_edits: bool = False,
    domain: str | None = None,
    orchestrate: bool = False,
) -> str:
    """以 headless 模式运行单个 agent 轮次并返回响应。

    参数:
        prompt: 要发送的用户消息。如果为 None，则从标准输入读取。
        allow_edits: 如果为 True（或通过 SMART_AGENT_ALLOW_EDITS 环境变量），
            自动批准文件编辑、命令和超出工作目录的访问。
            对于 headless 修改文件是必需的（编辑通常需要 TTY 批准）。
        domain: 本次运行使用的固定 Domain，必须显式为 code 或 career。
        orchestrate: 是否强制启用外层 Root Supervisor；默认按配置决定。

    返回:
        助手的响应文本。
    """
    from smart_agent.agent_loop import run_agent_turn
    from smart_agent.config import load_runtime_config
    from smart_agent.domains import assemble_capabilities, require_explicit_domain
    from smart_agent.logging_config import get_logger, setup_logging, structured_logging_requested
    from smart_agent.memory import MemoryManager
    from smart_agent.model_registry import create_model_adapter
    from smart_agent.permissions import PermissionManager
    from smart_agent.prompt import build_system_prompt
    from smart_agent.tools import create_default_tool_registry

    setup_logging(
        level=os.environ.get("SMART_AGENT_LOG_LEVEL", "WARNING"),
        structured=structured_logging_requested(),
    )
    logger = get_logger("headless")

    # Read prompt from stdin if not provided
    if prompt is None:
        if not sys.stdin.isatty():
            prompt = sys.stdin.read().strip()
        else:
            print("Usage: python -m smart_agent.headless <prompt>", file=sys.stderr)
            sys.exit(1)

    if not prompt:
        print("Error: empty prompt", file=sys.stderr)
        sys.exit(1)

    cwd = str(Path.cwd())
    selected_domain = require_explicit_domain(domain)

    # Load config
    try:
        runtime = load_runtime_config(cwd)
    except Exception as exc:  # noqa: BLE001
        # Persist the failure to the log file (issue #5), not just stderr.
        logger.error("Config load failed: %s", exc, exc_info=True)
        print(f"Config error: {exc}", file=sys.stderr)
        sys.exit(1)

    runtime = {
        **runtime,
        "activeDomain": selected_domain.value,
        "domainStage": "default",
    }
    from smart_agent.agents import decide_supervision

    supervisor_mode = runtime.get("supervisorMode", runtime.get("orchestratorMode", "off"))
    supervisor_decision = decide_supervision(prompt, supervisor_mode, force=orchestrate)
    runtime["supervisorAdmission"] = supervisor_decision.to_dict()
    use_orchestrator = supervisor_decision.should_supervise
    domain_capabilities = assemble_capabilities(selected_domain)

    # Initialize components
    tools = create_default_tool_registry(cwd, runtime=runtime)
    auto_approve = _allow_edits_requested(cli_flag=allow_edits)
    if auto_approve:
        logger.warning(
            "Headless --allow-edits is active: file edits, commands, and "
            "out-of-cwd access will be auto-approved for this run "
            "(non-interactive CI mode; approvals are session-scoped)."
        )
    permissions = PermissionManager(cwd, prompt=_make_auto_approve_prompt() if auto_approve else None)
    memory_mgr = MemoryManager(
        project_root=Path(cwd),
        namespace=domain_capabilities.memory_namespace,
    )

    trace_output_path = os.environ.get("SMART_AGENT_HEADLESS_MESSAGES_OUT", "").strip() or None
    if use_orchestrator:
        from smart_agent.agents import ModelRootPlanner, RootOrchestrator, SubagentRunner
        from smart_agent.tooling import ToolRegistry

        try:
            max_replans = max(
                0,
                int(runtime.get("supervisorMaxReplans", runtime.get("rootMaxReplans", 1))),
            )
        except (TypeError, ValueError):
            max_replans = 1
        root_session_id = f"root-{uuid.uuid4().hex[:12]}"
        try:
            root_model_name = str(
                runtime.get("supervisorModel")
                or runtime.get("rootModel")
                or runtime.get("model", "")
            ).strip()
            root_runtime = {
                **runtime,
                "model": root_model_name,
                "configuredModel": root_model_name,
                "isRootAgent": True,
            }
            root_model = create_model_adapter(
                model=root_runtime.get("model", ""),
                tools=ToolRegistry([]),
                runtime=root_runtime,
            )
            result = RootOrchestrator(
                runner=SubagentRunner(
                    cwd=cwd,
                    runtime=runtime,
                    tools=tools,
                    permissions=permissions,
                ),
                planner=ModelRootPlanner(root_model),
                max_replans=max_replans,
            ).run(
                prompt,
                domain=selected_domain,
                parent_session_id=root_session_id,
            )
            output = result.output_artifact.content.get("output", {}) if result.output_artifact else {}
            response_text = str(output.get("text") or "").strip() if isinstance(output, dict) else ""
            if not response_text:
                response_text = result.error or f"Root Agent ended with status={result.status}."
            if trace_output_path:
                path = Path(trace_output_path)
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(
                    json.dumps(
                        {
                            "cwd": cwd,
                            "prompt": prompt,
                            "model": runtime.get("model"),
                            "root_model": root_model_name,
                            "orchestrated": True,
                            "supervisor_decision": supervisor_decision.to_dict(),
                            "root_result": result.to_dict(),
                            "assistant_response": response_text,
                        },
                        ensure_ascii=False,
                        indent=2,
                        default=str,
                    ),
                    encoding="utf-8",
                )
            return response_text
        except Exception as exc:  # noqa: BLE001
            logger.error("Root orchestration error: %s", exc)
            if trace_output_path:
                path = Path(trace_output_path)
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(
                    json.dumps(
                        {
                            "cwd": cwd,
                            "prompt": prompt,
                            "model": runtime.get("model"),
                            "orchestrated": True,
                            "supervisor_decision": supervisor_decision.to_dict(),
                            "assistant_response": f"Error: {exc}",
                            "error": str(exc),
                        },
                        ensure_ascii=False,
                        indent=2,
                        default=str,
                    ),
                    encoding="utf-8",
                )
            return f"Error: {exc}"
        finally:
            try:
                tools.dispose()
            except Exception:  # noqa: BLE001
                pass

    model = create_model_adapter(
        model=runtime.get("model", ""),
        tools=tools,
        runtime=runtime,
    )

    messages = [
        {
            "role": "system",
            "content": build_system_prompt(
                cwd,
                permissions.get_summary(),
                {
                    "skills": tools.get_skills(),
                    "mcpServers": tools.get_mcp_servers(),
                    "memory_context": memory_mgr.get_relevant_context(),
                    "active_domain": selected_domain.value,
                    "domain_prompt": domain_capabilities.system_prompt,
                    "runtime": runtime,
                },
            ),
        },
        {"role": "user", "content": prompt},
    ]

    logger.info("Headless run: %s", prompt[:80])
    try:
        result_messages = run_agent_turn(
            model=model,
            tools=tools,
            messages=messages,
            cwd=cwd,
            permissions=permissions,
            runtime=runtime,
        )

        # Extract last assistant message
        last_assistant = next(
            (m for m in reversed(result_messages) if m["role"] == "assistant"),
            None,
        )
        response_text = last_assistant["content"] if last_assistant else "(no response)"
        _write_headless_messages_trace(
            trace_output_path,
            cwd=cwd,
            prompt=prompt,
            runtime=runtime,
            result_messages=result_messages,
            response_text=response_text,
        )
        return response_text

    except Exception as exc:  # noqa: BLE001
        logger.error("Headless error: %s", exc)
        response_text = f"Error: {exc}"
        _write_headless_messages_trace(
            trace_output_path,
            cwd=cwd,
            prompt=prompt,
            runtime=runtime,
            result_messages=[],
            response_text=response_text,
            error_text=str(exc),
        )
        return response_text
    finally:
        try:
            tools.dispose()
        except Exception:  # noqa: BLE001
            pass


def main() -> None:
    """Headless 模式的 CLI 入口点。

    解析命令行参数，提取 --allow-edits 标志，将剩余参数拼接为提示词，
    调用 run_headless 执行并打印结果。

    返回:
        None
    """
    parser = argparse.ArgumentParser(description="Run one Smart Agent turn.")
    parser.add_argument("--domain", required=True, choices=["code", "career"])
    parser.add_argument("--allow-edits", action="store_true")
    parser.add_argument(
        "--supervise",
        "--orchestrate",
        dest="orchestrate",
        action="store_true",
        help="Force the Root Supervisor for this task (legacy alias: --orchestrate).",
    )
    parser.add_argument("prompt", nargs="*")
    args = parser.parse_args()
    prompt = " ".join(args.prompt) if args.prompt else None
    response = run_headless(
        prompt,
        allow_edits=args.allow_edits,
        domain=args.domain,
        orchestrate=args.orchestrate,
    )
    print(response)


if __name__ == "__main__":
    main()
