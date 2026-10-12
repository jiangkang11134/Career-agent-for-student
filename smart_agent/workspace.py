"""工作区工具路径解析模块。

提供工具路径的解析和权限检查功能，确保文件操作
不会超出工作区边界。
"""

from __future__ import annotations

from pathlib import Path

from smart_agent.tooling import ToolContext


def resolve_tool_path(context: ToolContext, input_path: str, intent: str) -> Path:
    """解析并验证工具操作的路径。

    将可能相对路径转换为绝对路径，并进行安全检查。
    如果存在权限管理器，则通过权限管理器进行访问控制；
    否则回退到检查路径是否在工作区内。

    参数:
        context: 工具上下文，包含当前工作目录和权限信息
        input_path: 输入的路径字符串（可以是相对路径或绝对路径）
        intent: 操作意图描述，用于权限判断

    返回:
        解析后的标准化绝对路径

    抛出:
        PermissionError: 路径转义了工作区边界且无权限管理器
    """
    workspace_root = Path(context.cwd).resolve()
    candidate = Path(str(input_path or "").strip())
    if not candidate.is_absolute():
        # Models sometimes echo the workspace directory name even though tool
        # paths are documented as workspace-relative (for example,
        # ``smart agent/.git`` while cwd is ``.../Smart Agent``).  Strip one
        # exact root-name prefix; this stays inside the same workspace and
        # avoids turning a recoverable naming mismatch into repeated failures.
        parts = candidate.parts
        if parts and parts[0].casefold() == workspace_root.name.casefold():
            candidate = Path(*parts[1:]) if len(parts) > 1 else Path(".")
        target = workspace_root / candidate
    else:
        target = candidate
    normalized = target.resolve()

    if context.permissions is not None:
        context.permissions.ensure_path_access(str(normalized), intent)
    else:
        # Fallback: block paths that escape the workspace when no permissions manager
        try:
            normalized.relative_to(workspace_root)
        except ValueError:
            raise PermissionError(f"Path escapes workspace: {input_path}")

    return normalized
