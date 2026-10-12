"""工具注册与初始化模块。

负责收集和注册 Smart Agent 的所有内置工具，
包括共享工具、Domain 专属工具以及 MCP 工具。
根据固定 Domain 和工作流阶段选择加载的工具集。
"""

from dataclasses import asdict

from smart_agent.domains import assemble_capabilities, require_explicit_domain
from smart_agent.mcp import create_mcp_backed_tools
from smart_agent.skills import discover_skills
from smart_agent.tooling import ToolRegistry
from smart_agent.tools.ask_user import ask_user_tool
from smart_agent.tools.batch_ops import batch_copy_tool, batch_delete_tool, batch_move_tool
from smart_agent.tools.career import CAREER_TOOLS
from smart_agent.tools.career_task import career_task_tool
from smart_agent.tools.code_nav import find_references_tool, find_symbols_tool, get_ast_info_tool
from smart_agent.tools.code_review import code_review_tool
from smart_agent.tools.complete_task import complete_task_tool
from smart_agent.tools.diff_viewer import diff_viewer_tool
from smart_agent.tools.edit_file import edit_file_tool
from smart_agent.tools.file_tree import file_tree_tool
from smart_agent.tools.git import git_tool
from smart_agent.tools.grep_files import grep_files_tool
from smart_agent.tools.list_files import list_files_tool
from smart_agent.tools.load_skill import create_load_skill_tool
from smart_agent.tools.patch_file import patch_file_tool
from smart_agent.tools.read_file import read_file_tool
from smart_agent.tools.run_command import run_command_tool
from smart_agent.tools.sandbox_test import sandbox_test_tool
from smart_agent.tools.task import task_tool
from smart_agent.tools.test_runner import test_runner_tool
from smart_agent.tools.todo_write import todo_write_tool
from smart_agent.tools.web_fetch import web_fetch_tool
from smart_agent.tools.web_search import web_search_tool
from smart_agent.tools.write_file import write_file_tool

_SHARED_TOOLS = [
    # User interaction
    ask_user_tool,
    # Web tools
    web_fetch_tool,
    web_search_tool,
    # Task management
    todo_write_tool,
    complete_task_tool,
]

_CODE_TOOLS = [
    # File operations
    list_files_tool,
    grep_files_tool,
    read_file_tool,
    write_file_tool,
    # modify_file_tool removed: identical to write_file (same _run/_validate)
    edit_file_tool,
    patch_file_tool,
    # Batch operations
    batch_copy_tool,
    batch_move_tool,
    batch_delete_tool,
    # Command execution
    run_command_tool,
    # Sub-agent
    task_tool,
    # Git workflow
    git_tool,
    # Code intelligence
    find_symbols_tool,
    find_references_tool,
    get_ast_info_tool,
    code_review_tool,
    # Visualization
    file_tree_tool,
    diff_viewer_tool,
    # Testing
    sandbox_test_tool,
    test_runner_tool,
]


def _mcp_servers_for_domain(runtime: dict, domain: str) -> dict:
    """Filter MCP servers by explicit domain; legacy untagged servers remain Code-only."""
    selected: dict = {}
    for name, config in dict(runtime.get("mcpServers", {})).items():
        if not isinstance(config, dict):
            continue
        declared = config.get("domains")
        if declared is None:
            if domain == "code":
                selected[name] = config
            continue
        if isinstance(declared, str):
            declared = [declared]
        allowed = {str(item).strip().lower() for item in declared if str(item).strip()}
        if domain in allowed:
            selected[name] = config
    return selected

def create_default_tool_registry(cwd: str, runtime: dict | None = None) -> ToolRegistry:
    """创建默认的工具注册表。

    合并共享工具、Domain 专属工具、技能加载工具和 MCP 工具，
    构成当前 Domain 与阶段可见的工具注册表。

    参数:
        cwd: 当前工作目录，用于发现技能工具
        runtime: 运行时配置字典（可选），包含 Domain、Stage 和 MCP 配置

    返回:
        配置好的 ToolRegistry 实例
    """
    runtime = runtime or {}
    domain = require_explicit_domain(
        runtime.get("activeDomain") or runtime.get("domain") or "code"
    )
    stage = str(runtime.get("domainStage") or "default")
    capabilities = assemble_capabilities(domain, stage)
    skills = [
        asdict(skill)
        for skill in discover_skills(
            cwd,
            agent_type=capabilities.skill_agent_type,
            stage=stage,
        )
    ]
    mcp = create_mcp_backed_tools(
        cwd=cwd,
        mcp_servers=_mcp_servers_for_domain(runtime, domain.value),
    )
    candidates = [*_SHARED_TOOLS, *_CODE_TOOLS, *CAREER_TOOLS, career_task_tool]
    tools = [tool for tool in candidates if tool.name in capabilities.tool_names]
    if runtime.get("readOnlyToolsOnly"):
        # Project explanations may inspect source, but must never mutate or
        # execute the workspace. Keep this allow-list explicit and bounded.
        inspect_tools = {
            "file_tree", "list_files", "grep_files", "read_file",
            "find_symbols", "find_references", "get_ast_info", "code_review",
            "diff_viewer",
        }
        tools = [tool for tool in tools if tool.name in inspect_tools]
    if domain.value == "career" and runtime.get("careerSingleAgent"):
        user_decision_tools = {
            "career_task",
            "confirm_career_facts",
            "reject_candidate_facts",
            "invalidate_career_facts",
        }
        tools = [tool for tool in tools if tool.name not in user_decision_tools]
    if not runtime.get("readOnlyToolsOnly"):
        tools.append(create_load_skill_tool(cwd, capabilities.skill_agent_type, stage))
    if not runtime.get("readOnlyToolsOnly"):
        tools.extend(mcp["tools"])
    return ToolRegistry(
        tools,
        skills=skills,
        mcp_servers=mcp["servers"],
        disposer=mcp["dispose"],
    )
